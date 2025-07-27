//////
/// Metadata parsing
//////

import * as vscode from "vscode";
import { showAlertAndPanic, log } from "./logging";
import * as yaml from "yaml";
import * as fs from "fs";
import * as path from "path";

const CONFIG_FILE_NAME = "delphyne.yaml";
const DEFAULT_RESULT_REFRESH_PERIOD_IN_SECONDS = 5;
const DEFAULT_STATUS_REFRESH_PERIOD_IN_SECONDS = 1;

/////
// Configuration File and Command Execution Contexts
/////

export interface CommandExecutionContext {
  modules?: string[];
  demo_files?: string[];
  strategy_dirs?: string[];
  prompt_dirs?: string[];
  data_dirs?: string[];
  cache_root?: string;
  result_refresh_period?: number | null;
  status_refresh_period?: number | null;
}

export interface Config extends CommandExecutionContext {}

export function getWorkspaceRoot(): string {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders) {
    showAlertAndPanic("No workspace is opened.");
  }
  return workspaceFolders[0].uri.fsPath;
}

export function loadConfig(): Config {
  const workspaceRoot = getWorkspaceRoot();
  const configFilePath = path.join(workspaceRoot, CONFIG_FILE_NAME);

  let config: Config = {};
  if (!fs.existsSync(configFilePath)) {
    log.info(
      "Config file 'delphyne.yaml' not found at the root of the workspace. Using the default configuration.",
    );
  } else {
    const configFileContent = fs.readFileSync(configFilePath, "utf8");
    config = yaml.parse(configFileContent);
  }
  if (config.result_refresh_period === undefined) {
    config.result_refresh_period = DEFAULT_RESULT_REFRESH_PERIOD_IN_SECONDS;
  }
  if (config.status_refresh_period === undefined) {
    config.status_refresh_period = DEFAULT_STATUS_REFRESH_PERIOD_IN_SECONDS;
  }
  return config;
}

export function getCommandExecutionContext(): CommandExecutionContext {
  return loadConfig();
}

//////
/// Demonstration Execution Contexts
//////

export interface ExecutionContext {
  strategy_dirs?: string[];
  modules?: string[];
}

export function getExecutionContext(): ExecutionContext {
  const config = loadConfig();
  return {
    strategy_dirs: config.strategy_dirs,
    modules: config.modules,
  };
}
