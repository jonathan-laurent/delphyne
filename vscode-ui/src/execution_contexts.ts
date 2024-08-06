//////
/// Metadata parsing
//////

import * as vscode from "vscode";
import { loadConfig } from "./config";

export interface ExecutionContext {
  strategy_dirs: string[];
  modules: string[];
}

interface FileMetadata {
  strategy_dirs: string[];
  modules: string[];
}

/** Extract FileMetadata from a YAML file. The metadata can be included in a block comment
 * such as the following:
 *  ```
 *  # @strategy_dirs: .
 *  # @modules: foo, bar
 *  ```
 * The block comment can only occur at the beginning of the file and be preceded by blank
 * lines or other comments.
 */
export function extractMetadata(fileContent: string): FileMetadata {
  const metadata: FileMetadata = { strategy_dirs: [], modules: [] };

  const lines = fileContent.split("\n");
  let metadataLines: string[] = [];
  for (const line of lines) {
    if (line.trim() === "") {
      continue;
    }
    if (line.trim().startsWith("#")) {
      metadataLines.push(line.trim());
    } else {
      break;
    }
  }

  for (const line of metadataLines) {
    const match = line.match(/@(\w+):(.*)/);
    if (match) {
      const key = match[1].trim();
      const value = match[2].trim();
      if (key === "strategy_dirs") {
        metadata.strategy_dirs = value.split(",").map((x) => x.trim());
      } else if (key === "modules") {
        metadata.modules = value.split(",").map((x) => x.trim());
      }
    }
  }

  return metadata;
}

export function getExecutionContextFromFileMetadata(
  document: vscode.TextDocument,
): ExecutionContext {
  const metadata = extractMetadata(document.getText());
  const strategy_dirs = metadata.strategy_dirs.map(
    (dir) => vscode.Uri.joinPath(document.uri, "..", dir).fsPath,
  );
  return { strategy_dirs, modules: metadata.modules };
}

export function getExecutionContext(): ExecutionContext {
  const config = loadConfig();
  return {
    strategy_dirs: config.strategy_dirs,
    modules: config.modules,
  };
}

/////
// Command execution contexts
/////

export interface CommandExecutionContext {
  base: ExecutionContext;
  demo_files: string[];
}

export function getCommandExecutionContext(): CommandExecutionContext {
  const config = loadConfig();
  return {
    base: { strategy_dirs: config.strategy_dirs, modules: config.modules },
    demo_files: config.demo_files,
  };
}
