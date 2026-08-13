//! Dev/test-only MCP automation server (Phase 9, see ROADMAP.md).
//!
//! Lets an agent/test harness drive `ManifoldApp` programmatically (select
//! objects, set transforms, import files, list state) over a local TCP
//! socket, without synthesizing pointer/keyboard input. Entirely gated
//! behind the `mcp-server` Cargo feature — never compiled into a release
//! binary built without that feature.
//!
//! Runs on its own dedicated tokio runtime inside a spawned `std::thread`;
//! `ManifoldApp`'s egui/wgpu event loop on the main thread has no tokio
//! involvement. Tool calls cross the thread boundary as `Command`s sent
//! over a `std::sync::mpsc::Sender`; `ManifoldApp::update` drains the
//! receiver once per frame and applies mutations on the UI thread, same as
//! any other state mutation. Query-style tools pair their `Command` with a
//! reply `Sender` so the async tool-call handler can wait for the next
//! frame to produce an answer.

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::service::serve_server;
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use tokio::net::TcpListener;

/// Address the automation server listens on. Local-only, fixed, dev-tool
/// port — not configurable yet since this never ships in release builds.
pub const ADDR: &str = "127.0.0.1:8931";

/// Commands sent from the MCP tool-call handler (async, on the server's
/// own tokio thread) to `ManifoldApp::update` (sync, on the UI thread).
pub enum Command {
    SelectObject(usize),
    SetTransform {
        index: usize,
        x: f64,
        y: f64,
        z: f64,
    },
    ImportFile(PathBuf),
    /// Query: reply with a JSON array describing every loaded object.
    ListObjects(Sender<String>),
    /// Query: reply with the currently selected object index, if any.
    GetSelected(Sender<Option<usize>>),
}

#[derive(Deserialize, JsonSchema)]
struct SelectObjectParams {
    index: usize,
}

#[derive(Deserialize, JsonSchema)]
struct SetTransformParams {
    index: usize,
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Deserialize, JsonSchema)]
struct ImportFileParams {
    path: String,
}

/// MCP tool surface. Holds only a `Sender<Command>` clone — all actual
/// scene state lives in `ManifoldApp` on the UI thread.
#[derive(Clone)]
pub struct SceneServer {
    tx: Sender<Command>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl SceneServer {
    fn new(tx: Sender<Command>) -> Self {
        Self {
            tx,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "List every object currently loaded in the scene, as JSON.")]
    async fn list_objects(&self) -> String {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        if self.tx.send(Command::ListObjects(reply_tx)).is_err() {
            return "[]".to_string();
        }
        reply_rx.recv().unwrap_or_else(|_| "[]".to_string())
    }

    #[tool(description = "Get the index of the currently selected object, if any.")]
    async fn get_selected(&self) -> String {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        if self.tx.send(Command::GetSelected(reply_tx)).is_err() {
            return "null".to_string();
        }
        match reply_rx.recv() {
            Ok(Some(index)) => index.to_string(),
            _ => "null".to_string(),
        }
    }

    #[tool(description = "Select an object by its index in the object list.")]
    async fn select_object(
        &self,
        Parameters(SelectObjectParams { index }): Parameters<SelectObjectParams>,
    ) -> String {
        let _ = self.tx.send(Command::SelectObject(index));
        format!("selected object {index}")
    }

    #[tool(description = "Set an object's translation (position) by index.")]
    async fn set_transform(
        &self,
        Parameters(SetTransformParams { index, x, y, z }): Parameters<SetTransformParams>,
    ) -> String {
        let _ = self.tx.send(Command::SetTransform { index, x, y, z });
        format!("object {index} moved to ({x}, {y}, {z})")
    }

    #[tool(description = "Import a mesh file (.stl or .3mf) from an absolute path.")]
    async fn import_file(
        &self,
        Parameters(ImportFileParams { path }): Parameters<ImportFileParams>,
    ) -> String {
        let _ = self.tx.send(Command::ImportFile(PathBuf::from(&path)));
        format!("importing {path}")
    }
}

#[tool_handler]
impl ServerHandler for SceneServer {}

async fn run_tcp_server(addr: SocketAddr, tx: Sender<Command>) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "MCP automation server listening");

    loop {
        let (stream, peer) = listener.accept().await?;
        tracing::debug!(%peer, "MCP automation client connected");
        let service = SceneServer::new(tx.clone());

        tokio::spawn(async move {
            let (read_half, write_half) = tokio::io::split(stream);
            match serve_server(service, (read_half, write_half)).await {
                Ok(running) => {
                    if let Err(error) = running.waiting().await {
                        tracing::warn!(%peer, ?error, "MCP session ended with error");
                    }
                }
                Err(error) => {
                    tracing::warn!(%peer, ?error, "failed to initialize MCP session");
                }
            }
        });
    }
}

/// Spawn the automation server on its own OS thread with a private tokio
/// runtime, and return a `Receiver<Command>` for `ManifoldApp` to drain
/// once per frame. The server thread runs independently of the egui/wgpu
/// main loop and is not joined — it is killed when the process exits.
pub fn spawn(addr: &str) -> anyhow::Result<Receiver<Command>> {
    let addr: SocketAddr = addr.parse()?;
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::Builder::new()
        .name("manifold-mcp".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    tracing::error!(?error, "failed to build MCP server tokio runtime");
                    return;
                }
            };
            if let Err(error) = runtime.block_on(run_tcp_server(addr, tx)) {
                tracing::error!(?error, "MCP automation server exited with error");
            }
        })?;

    Ok(rx)
}
