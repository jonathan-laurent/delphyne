//////
/// Configuration
//////

import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import * as yaml from "yaml";
import { showAlertAndPanic } from "./logging";

interface Config {
  strategy_dirs: string[];
  prompt_dirs: string[];
  data_dirs?: string[];
  modules: string[];
  demo_files: string[];
  result_refresh_period?: number | null;
  status_refresh_period?: number | null;
}

export function loadConfig(): Config {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders) {
    showAlertAndPanic("No workspace is opened.");
  }

  const workspaceRoot = workspaceFolders[0].uri.fsPath;
  const configFilePath = path.join(workspaceRoot, "delphyne.yaml");

  if (!fs.existsSync(configFilePath)) {
    showAlertAndPanic(
      `Config file 'delphyne.yaml' not found at the root of the workspace.`,
    );
  }

  const configFileContent = fs.readFileSync(configFilePath, "utf8");
  const config: Config = yaml.parse(configFileContent) as Config;

  // Convert strategy_dirs, prompt_dirs and data_dirs to absolute paths
  config.strategy_dirs = config.strategy_dirs.map((dir) =>
    path.resolve(workspaceRoot, dir),
  );
  config.prompt_dirs = config.prompt_dirs.map((dir) =>
    path.resolve(workspaceRoot, dir),
  );
  if (config.data_dirs) {
    config.data_dirs = config.data_dirs.map((dir) =>
      path.resolve(workspaceRoot, dir),
    );
  }
  // Convert demo_files to absolute paths with .demo.yaml extension and check existence
  config.demo_files = config.demo_files.map((file) => {
    const demoFilePath = path.resolve(workspaceRoot, `${file}.demo.yaml`);
    if (!fs.existsSync(demoFilePath)) {
      showAlertAndPanic(`Demo file '${demoFilePath}' does not exist.`);
    }
    return demoFilePath;
  });

  return config;
}
