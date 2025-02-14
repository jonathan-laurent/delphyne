//////
/// Interaction with the Delphyne Language Server
//////

import * as vscode from "vscode";
import { log, serverLogChannel } from "./logging";
import { spawn, ChildProcessWithoutNullStreams } from "child_process";
import {
  Task,
  TasksManager,
  MessageStream,
  TaskOutcome,
  isValidTaskMessage,
  TaskInfo,
  TaskOptions,
} from "./tasks";
import { waitFor } from "./utils";
import { ReadableStreamReadResult } from "stream/web";

const CANCEL_POLLING_INTERVAL = 100;
const SERVER_ADDRESS = "http://localhost:8000";

//////
/// Server wrapper
//////

export class DelphyneServer {
  private tasksManager: TasksManager;
  private nextSimpleQueryId = 0;
  constructor(
    context: vscode.ExtensionContext,
    private process: ChildProcessWithoutNullStreams | null,
  ) {
    this.tasksManager = new TasksManager(context);
  }

  kill() {
    log.info("Terminating the Delphyne server.");
    this.process?.kill();
  }

  async callServer(query: string, payload: any) {
    const address = `${SERVER_ADDRESS}/${query}`;
    const config = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    };
    return await fetch(address, config);
  }

  async query(query: string, payload: any): Promise<any> {
    const id = this.nextSimpleQueryId++;
    try {
      log.debug(`Sending Query #${id}:`, { query, payload });
      const response = await this.callServer(query, payload);
      const json = await response.json();
      log.debug(`Received answer for Query #${id}:`, json);
      return json;
    } catch (error) {
      log.error("Server error:", error);
      throw error;
    }
  }

  async *streamingQuery<T>(
    query: string,
    payload: any,
    token: vscode.CancellationToken,
  ): MessageStream<T> {
    const response = await this.callServer(query, payload);
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Failed to read the response body.");
    }
    let buffer = "";
    try {
      let readNext: () => Promise<ReadableStreamReadResult<any>> = () =>
        reader.read();
      while (true) {
        const next = await waitFor(readNext(), CANCEL_POLLING_INTERVAL);
        if (next[0] === "timeout") {
          // If no message is received for a while
          if (token.isCancellationRequested) {
            reader.cancel();
            break;
          } else {
            readNext = () => next[1];
            continue;
          }
        }
        readNext = () => reader.read();
        const result = next[1];
        if (result.done) {
          if (buffer !== "") {
            if (response.status === 422) {
              log.error("Malformed query (error 422)", {
                error: JSON.parse(buffer),
              });
            } else {
              log.error("Unexpected end of stream:", {
                query,
                payload,
                buffer,
              });
            }
          }
          break;
        }
        buffer += new TextDecoder().decode(result.value, { stream: true });
        const parts = buffer.split("\n\n");

        for (let i = 0; i < parts.length - 1; i++) {
          const message: unknown = JSON.parse(parts[i]);
          if (!isValidTaskMessage(message)) {
            log.error("Invalid message received from the server:", message);
            continue;
          }
          log.debug("msg", message);
          yield message;
        }
        buffer = parts[parts.length - 1];
      }
    } catch (e) {
      log.error("Exception raised while executing a streaming query:", {
        query,
        payload,
      });
    } finally {
      reader.cancel();
    }
  }

  // A task is associated to an element and is cancelled if the element disappears
  launchTask(
    info: TaskInfo,
    options: TaskOptions<any>,
    query: string,
    payload: any,
  ): { outcome: Promise<TaskOutcome<any>>; cancel: () => void } {
    const id = this.tasksManager.freshTaskId();
    log.debug(`Launching Task #${id}:`, { info, query, payload });
    const stream = (token: vscode.CancellationToken) =>
      this.streamingQuery(query, payload, token);
    const task = new Task(id, info, options, stream);
    const genOutcome = async () => {
      await this.tasksManager.run(task);
      const outcome = task.outcome();
      const verb = outcome.cancelled ? "cancelled" : "completed";
      log.debug(`Task #${id} ${verb}:`, outcome.result);
      return outcome;
    };
    const cancel = () => task.cancel();
    return { outcome: genOutcome(), cancel };
  }
}

//////
/// Starting the server process
//////

export async function startServer(
  context: vscode.ExtensionContext,
): Promise<DelphyneServer | null> {
  let process: ChildProcessWithoutNullStreams | null = null;
  const running = await isServerAlreadyRunning();
  if (running) {
    log.info("The Delphyne server is already running outside VSCode.");
  } else {
    const python = await findPythonCommand();
    const process = await startServerProcess(python);
    if (!process) {
      return null;
    }
  }
  return new DelphyneServer(context, process);
}

// Determine whether the Delphyne server is runnig at the expected address.
async function isServerAlreadyRunning(): Promise<boolean> {
  try {
    log.debug("Pinging the Delphyne server...");
    const response = await fetch(SERVER_ADDRESS + "/ping-delphyne");
    return response.ok;
  } catch (error) {
    log.debug("Failed to ping the Delphyne server:", error);
    return false;
  }
}

async function findPythonCommand(): Promise<string> {
  const pythonExtension = vscode.extensions.getExtension("ms-python.python");
  if (pythonExtension) {
    if (!pythonExtension.isActive) {
      log.debug("Activating the Python extension.");
      await pythonExtension.activate();
    }
    const pythonAPI = pythonExtension.exports;
    const pythonPath =
      await pythonAPI.settings.getExecutionDetails().execCommand[0];
    log.info("Using the Python environment at", pythonPath);
    return pythonPath;
  } else {
    log.warn(
      "Python VSCode extension not found. " +
        "Attempting to use the `python` command to launch the server.",
    );
    return "python";
  }
}

async function startServerProcess(
  python: string,
): Promise<ChildProcessWithoutNullStreams | null> {
  return new Promise((resolve) => {
    const server = spawn(python, ["-m", "delphyne.server"]);
    server.stdout.on("data", (data) => {
      serverLogChannel?.append(`${data}`);
    });
    server.stderr.on("data", (data) => {
      serverLogChannel?.append(`${data}`);
      // This string is inserted by Uvicorn
      if (data.toString().includes("Application startup complete.")) {
        log.info("Delphyne server started successfully at", SERVER_ADDRESS);
        resolve(server);
      }
    });
    server.on("close", (code) => {
      log.error(`Delphyne server exited with code ${code}`);
      resolve(null);
    });
  });
}
