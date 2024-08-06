//////
/// Configuration
//////

import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import * as yaml from "yaml";

interface Config {
  strategy_dirs: string[];
  modules: string[];
  demo_files: string[];
}

export function loadConfig(): Config {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders) {
    throw new Error("No workspace is opened.");
  }

  const workspaceRoot = workspaceFolders[0].uri.fsPath;
  const configFilePath = path.join(workspaceRoot, "delphyne.yaml");

  if (!fs.existsSync(configFilePath)) {
    throw new Error(
      `Config file 'delphyne.yaml' not found at the root of the workspace.`,
    );
  }

  const configFileContent = fs.readFileSync(configFilePath, "utf8");
  const config: Config = yaml.parse(configFileContent) as Config;

  // Convert strategy_dirs to absolute paths
  config.strategy_dirs = config.strategy_dirs.map((dir) =>
    path.resolve(workspaceRoot, dir),
  );

  // Convert demo_files to absolute paths with .demo.yaml extension and check existence
  config.demo_files = config.demo_files.map((file) => {
    const demoFilePath = path.resolve(workspaceRoot, `${file}.demo.yaml`);
    if (!fs.existsSync(demoFilePath)) {
      throw new Error(`Demo file '${demoFilePath}' does not exist.`);
    }
    return demoFilePath;
  });

  return config;
}
