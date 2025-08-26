//////
/// Metadata parsing
//////

import * as vscode from "vscode";
import { showAlertAndPanic, log, showAlert } from "./logging";
import * as yaml from "yaml";
import * as fs from "fs";
import * as path from "path";

const CONFIG_FILE_NAME = "delphyne.yaml";
const DEFAULT_RESULT_REFRESH_PERIOD_IN_SECONDS = 5;
const DEFAULT_STATUS_REFRESH_PERIOD_IN_SECONDS = 1;

/////
// Workspace Detection
/////

/**
 * Find the project root directory by looking for the delphyne.yaml file.
 * Searches up the directory tree from the starting directory.
 */
export function findRootDir(startingDir: string): string | null {
  let currentDir = path.resolve(startingDir);
  const root = path.parse(currentDir).root;
  while (currentDir !== root) {
    const configPath = path.join(currentDir, CONFIG_FILE_NAME);
    if (fs.existsSync(configPath)) {
      return currentDir;
    }
    currentDir = path.dirname(currentDir);
  }
  return null;
}

function getGlobalWorkspaceRoot(): string {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders) {
    showAlertAndPanic("No workspace is opened.");
  }
  return workspaceFolders[0].uri.fsPath;
}

export function getWorkspaceRoot(filePath: string | null): string {
  let dir: string;
  if (!filePath) {
    dir = getGlobalWorkspaceRoot();
  } else {
    dir = findRootDir(filePath) ?? getGlobalWorkspaceRoot();
  }
  // log.info("Using root directory: " + dir); // A bit too noisy...
  return dir;
}

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

function loadConfig(rootDir: string): Config {
  const configFilePath = path.join(rootDir, CONFIG_FILE_NAME);
  let config: Config = {};
  if (!fs.existsSync(configFilePath)) {
    log.info(
      "Config file 'delphyne.yaml' not found at the root of the workspace. Using the default configuration.",
    );
  } else {
    const configFileContent = fs.readFileSync(configFilePath, "utf8");
    config = yaml.parse(configFileContent);
    if (!config) {
      config = {}; // An empty YAML file will be parsed as null, so we default to an empty object
    }
  }
  if (config.result_refresh_period === undefined) {
    config.result_refresh_period = DEFAULT_RESULT_REFRESH_PERIOD_IN_SECONDS;
  }
  if (config.status_refresh_period === undefined) {
    config.status_refresh_period = DEFAULT_STATUS_REFRESH_PERIOD_IN_SECONDS;
  }
  return config;
}

// Load the local configuration for a file by FIRST loading the global
// configuration and then looking for a configuration comment block within the
// document (`extractConfigBlock`), parsing it as YAML and updating the global
// configuration with all new fields. If the configuration block exists but is
// not valid YAML, raise an error.
export function loadLocalConfig(doc: vscode.TextDocument): Config {
  // Start with the global config
  const rootDir = getWorkspaceRoot(doc.fileName);
  const globalConfig = loadConfig(rootDir);
  const text = doc.getText();
  const configBlock = extractConfigBlock(text);
  if (!configBlock) {
    // No local config block, return global config
    return globalConfig;
  }
  const ignoreMsg = "Ignoring the local configuration block.";
  let localConfig: Partial<Config> = {};
  try {
    const parsed = yaml.parse(configBlock) || {};
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed) ||
      Object.getOwnPropertyNames(parsed).some((k) => typeof k !== "string")
    ) {
      showAlert(
        `Local configuration block must be a YAML dictionary (mapping) with string keys. ${ignoreMsg}`,
      );
    }
    localConfig = parsed;
  } catch (e) {
    showAlert(`Invalid YAML in local configuration block. ${ignoreMsg}`);
    log.error(`Invalid YAML in local configuration block`, e);
  }
  // Merge global config with local config (local overrides global)
  return { ...globalConfig, ...localConfig };
}

// Returns the command execution context for a specific document, using local config if present
export function getLocalCommandExecutionContext(
  doc: vscode.TextDocument,
): CommandExecutionContext {
  return loadLocalConfig(doc);
}

//////
/// Demonstration Execution Contexts
//////

export interface ExecutionContext {
  strategy_dirs?: string[];
  modules?: string[];
}

function executionContextOfConfig(config: Config): ExecutionContext {
  return {
    // By default, look for strategies in the workspace root directory. This is
    // consistent with the defaults of `CommandExecutionContext`.
    strategy_dirs: config.strategy_dirs ?? ["."],
    modules: config.modules ?? [],
  };
}

export function getLocalExecutionContext(
  doc: vscode.TextDocument,
): ExecutionContext {
  const config = loadLocalConfig(doc);
  return executionContextOfConfig(config);
}

//////
// Local Config Updates
//////

// Search for a comment block such as this and return its content as a string.
//
//  # @config
//  # ...
//  # ...
//  # @end
//
// The block must be at the start of the document: only blank lines and comments are allowed before it.
// The returned string does not include the `#` characters or the `@config` and `@end` markers.
export function extractConfigBlock(document: string): string | null {
  const lines = document.split("\n");
  let configStartIndex = -1;
  let configEndIndex = -1;

  // Find the start of the config block
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    // If we encounter a non-comment, non-blank line before finding @config, return null
    if (line && !line.startsWith("#")) {
      return null;
    }
    // Check if this is the start of the config block
    if (line === "# @config") {
      configStartIndex = i;
      break;
    }
  }
  // If no @config found, return null
  if (configStartIndex === -1) {
    return null;
  }

  // Find the end of the config block and validate content
  for (let i = configStartIndex + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "# @end") {
      configEndIndex = i;
      break;
    }
    // Validate that each line is a comment or blank
    if (line && !line.startsWith("#")) {
      return null;
    }
  }
  // If no @end found, return null
  if (configEndIndex === -1) {
    return null;
  }

  // Extract the content between @config and @end
  const configLines = [];
  for (let i = configStartIndex + 1; i < configEndIndex; i++) {
    const line = lines[i];
    if (line.startsWith("# ")) {
      configLines.push(line.substring(2));
    } else if (line.trim() === "#") {
      configLines.push("");
    } else {
      configLines.push(line);
    }
  }

  return configLines.join("\n");
}
