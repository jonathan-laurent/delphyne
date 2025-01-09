//////
/// Logging utilities
//////

import * as vscode from "vscode";

export let log: vscode.LogOutputChannel;
export let serverLogChannel: vscode.OutputChannel | null = null;

export function initLogChannels() {
  if (!log) {
    log = vscode.window.createOutputChannel("Delphyne", { log: true });
  }
  if (serverLogChannel === null) {
    serverLogChannel = vscode.window.createOutputChannel("Delphyne Server");
  }
}

export function showAlert(message: string) {
  log.error(message);
  vscode.window.showErrorMessage(message, "Show Log").then((value) => {
    if (value === "Show Log") {
      log.show();
    }
  });
}

export function showAlertAndPanic(message: string): never {
  showAlert(message);
  throw new Error(message);
}
