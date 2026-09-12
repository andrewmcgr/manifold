use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::{accept_async, tungstenite::Message, WebSocketStream};

pub type Socket = WebSocketStream<TcpStream>;
pub struct Request {
    pub path: String,
    pub respond: oneshot::Sender<Value>,
}
pub struct Server {
    pub url: String,
    pub sockets: mpsc::UnboundedReceiver<Socket>,
    pub requests: mpsc::UnboundedReceiver<Request>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Server {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (ws_tx, sockets) = mpsc::unbounded_channel();
        let (http_tx, requests) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let mut handlers = tokio::task::JoinSet::new();
            loop {
                let (mut tcp, _) = listener.accept().await.unwrap();
                let ws_tx = ws_tx.clone();
                let http_tx = http_tx.clone();
                handlers.spawn(async move {
                    let mut peek=[0u8;2048];
                    loop {
                        let n=tcp.peek(&mut peek).await.unwrap();
                        if n==0 {return;}
                        if String::from_utf8_lossy(&peek[..n]).contains("\r\n\r\n") {break;}
                        tokio::task::yield_now().await;
                    }
                    if String::from_utf8_lossy(&peek).to_lowercase().contains("upgrade: websocket") {
                        if let Ok(ws)=accept_async(tcp).await {let _=ws_tx.send(ws);} return;
                    }
                    let mut bytes=Vec::new();let mut buf=[0u8;8192];
                    let header_end=loop {
                        let n=tcp.read(&mut buf).await.unwrap();if n==0 {return;}
                        bytes.extend_from_slice(&buf[..n]);
                        if let Some(p)=bytes.windows(4).position(|s|s==b"\r\n\r\n") {break p+4;}
                    };
                    let header=String::from_utf8_lossy(&bytes[..header_end]);
                    let path=header.split_whitespace().nth(1).unwrap().to_string();
                    let length=header.lines().find_map(|line|line.to_lowercase().strip_prefix("content-length: ").and_then(|v|v.parse::<usize>().ok())).unwrap_or(0);
                    while bytes.len()<header_end+length {
                        let n=tcp.read(&mut buf).await.unwrap();if n==0 {return;}bytes.extend_from_slice(&buf[..n]);
                    }
                    let (respond,reply)=oneshot::channel();
                    if http_tx.send(Request{path,respond}).is_err(){return;}
                    if let Ok(value)=reply.await {
                        let body=value.to_string();
                        let response=format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body);
                        let _=tcp.write_all(response.as_bytes()).await;
                    }
                });
            }
        });
        Self {
            url,
            sockets,
            requests,
            task,
        }
    }
    pub async fn socket(&mut self) -> Socket {
        tokio::time::timeout(Duration::from_secs(3), self.sockets.recv())
            .await
            .unwrap()
            .unwrap()
    }
    pub async fn request(&mut self) -> Request {
        tokio::time::timeout(Duration::from_secs(3), self.requests.recv())
            .await
            .unwrap()
            .unwrap()
    }
}
pub async fn rpc(socket: &mut Socket) -> Value {
    let msg = tokio::time::timeout(Duration::from_secs(3), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    serde_json::from_str(msg.to_text().unwrap()).unwrap()
}
pub async fn reply(socket: &mut Socket, id: &Value, result: Value) {
    socket
        .send(Message::Text(json!({"id":id,"result":result}).to_string()))
        .await
        .unwrap();
}
pub async fn subscribe(socket: &mut Socket, status: Value) {
    let identify = rpc(socket).await;
    assert_eq!(identify["method"], "server.connection.identify");
    reply(socket, &identify["id"], json!({"connection_id":1})).await;
    let info = rpc(socket).await;
    assert_eq!(info["method"], "server.info");
    reply(socket, &info["id"], json!({"klippy_state":"ready"})).await;
    let sub = rpc(socket).await;
    assert_eq!(sub["method"], "printer.objects.subscribe");
    reply(socket, &sub["id"], json!({"status":status})).await;
}
pub async fn until(mut predicate: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !predicate() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
