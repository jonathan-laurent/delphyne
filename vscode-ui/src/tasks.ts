//////
/// Scheduling Tasks in the Background
//////

import * as vscode from "vscode";
import { Element } from "./elements";
import { delay } from "./utils";
import { log } from "./logging";

const STATUS_BAR_REFRESH_INTERVAL = 500;

export type TaskMessage<T> =
  | ["log", string]
  | ["set_status", string]
  | ["set_result", T]
  | ["internal_error", string];

export type MessageStream<T> = AsyncGenerator<TaskMessage<T>, void, unknown>;

export function isValidTaskMessage(obj: unknown): obj is TaskMessage<any> {
  return (
    Array.isArray(obj) &&
    obj.length === 2 &&
    (obj[0] === "log" ||
      obj[0] === "set_status" ||
      obj[0] === "set_result" ||
      obj[0] === "internal_error")
  );
}

type TaskId = number;

// The result can be null if the task never set a result or was cancelled before doing so.
export interface TaskOutcome<T> {
  result: T | null;
  cancelled: boolean;
}

export type TaskInfo = {
  name: string;
  arg: string;
  origin: Element | null;
};

export type TaskOptions<T> = {
  sync_by_default?: boolean;
  sync_period_ms?: number;
  handle_result?: (result: T) => Promise<void>;
};

export class Task<T> {
  constructor(
    public id: TaskId,
    public info: TaskInfo,
    public options: TaskOptions<T>,
    public task: (token: vscode.CancellationToken) => MessageStream<T>,
  ) {
    if (this.options.sync_period_ms !== undefined) {
      this.sync = this.options.sync_by_default ?? false;
    }
  }
  public cancelled: boolean = false;
  public notify: boolean = false;
  public sync: boolean | null = null;
  public in_status_bar: boolean = false;
  public status_message: string = "";
  public creationDate: Date = new Date();
  public result: T | null = null;
  public cancellation = new vscode.CancellationTokenSource();
  private lastSyncDate: Date = new Date();

  outcome(): TaskOutcome<T> {
    return {
      result: this.result,
      cancelled: this.cancelled,
    };
  }

  contextValue(): string {
    // Format: task:notify-{true,false}:sync-{null,true,false}
    return `task:status-${this.in_status_bar}:notify-${this.notify}:sync-${this.sync}`;
  }

  async updateSource(): Promise<void> {
    if (this.result !== null && this.options.handle_result !== undefined) {
      await this.options.handle_result(this.result);
      this.lastSyncDate = new Date();
    }
  }

  async maybeUpdateSource(): Promise<void> {
    if (
      this.sync &&
      this.options.sync_period_ms !== undefined &&
      new Date().getTime() - this.lastSyncDate.getTime() >
        this.options.sync_period_ms
    ) {
      this.updateSource();
    }
  }

  cancel() {
    this.cancelled = true;
    this.cancellation.cancel();
  }
}

//////
/// Tasks Manager
//////

export class TasksManager {
  constructor(private context: vscode.ExtensionContext) {
    context.subscriptions.push(this.tasksView);
    this.registerCommands(context);
    this.refreshView();
  }

  private nextTaskId: TaskId = 0;
  public tasks: Task<any>[] = [];
  private tasksViewProvider: TasksViewProvider = new TasksViewProvider(this);
  private tasksView = vscode.window.createTreeView<Task<any>>(
    "delphyne.tasks",
    { treeDataProvider: this.tasksViewProvider },
  );
  public tasksLogChannel = vscode.window.createOutputChannel(
    "Delphyne Tasks Log",
    { log: true },
  );

  freshTaskId(): TaskId {
    return this.nextTaskId++;
  }

  refreshView() {
    this.tasksViewProvider.refresh();
  }

  removeTask(task: Task<any>) {
    task.in_status_bar = false;
    this.tasks = this.tasks.filter((t) => t.id !== task.id);
    if (!task.cancelled && task.notify) {
      // Compute total task duration and format it nicely to add to `info`
      const duration = formatDuration(task.creationDate, new Date());
      const info = `duration: ${duration}`;
      const tinfo = task.info;
      const message = `Completed Task ${task.id}: "${tinfo.name} ${tinfo.arg}" (${info})`;
      vscode.window.showInformationMessage(message);
    }
  }

  registerCommands(context: vscode.ExtensionContext) {
    const cancelTask = (task: Task<any>) => {
      task.cancel();
    };
    const viewTaskInfo = async (task: Task<any>) => {
      task.in_status_bar = true;
      this.tasksViewProvider.refresh();
      const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        0,
      );
      this.context.subscriptions.push(statusBarItem);
      while (task.in_status_bar) {
        statusBarItem.text =
          "$(server-process) " +
          `${task.info.name} ${task.info.arg}: ${task.status_message}`;
        statusBarItem.show();
        await delay(STATUS_BAR_REFRESH_INTERVAL);
      }
      statusBarItem.dispose();
    };
    const hideTaskInfo = (task: Task<any>) => {
      task.in_status_bar = false;
      this.tasksViewProvider.refresh();
    };
    const notifyTaskCompleted = (task: Task<any>) => {
      task.notify = true;
      this.tasksViewProvider.refresh();
    };
    const undoNotifyTaskCompleted = (task: Task<any>) => {
      task.notify = false;
      this.tasksViewProvider.refresh();
    };
    const updateSource = async (task: Task<any>) => {
      await task.updateSource();
    };
    vscode.commands.registerCommand("delphyne.viewTaskInfo", viewTaskInfo);
    vscode.commands.registerCommand("delphyne.hideTaskInfo", hideTaskInfo);
    vscode.commands.registerCommand(
      "delphyne.notifyTaskCompleted",
      notifyTaskCompleted,
    );
    vscode.commands.registerCommand(
      "delphyne.undoNotifyTaskCompleted",
      undoNotifyTaskCompleted,
    );
    vscode.commands.registerCommand("delphyne.cancelTask", cancelTask);
    vscode.commands.registerCommand("delphyne.updateTaskSource", updateSource);
    vscode.commands.registerCommand(
      "delphyne.enableTaskSync",
      (task: Task<any>) => {
        task.sync = true;
        this.tasksViewProvider.refresh();
      },
    );
    vscode.commands.registerCommand(
      "delphyne.disableTaskSync",
      (task: Task<any>) => {
        task.sync = false;
        this.tasksViewProvider.refresh();
      },
    );
  }

  async run<T>(task: Task<T>): Promise<void> {
    this.tasks.push(task);
    this.refreshView();
    const stream = task.task(task.cancellation.token);
    try {
      for await (const [kind, data] of stream) {
        if (kind === "set_status") {
          if (data) {
            task.status_message = data;
          }
        } else if (kind === "set_result") {
          task.result = data;
          await task.maybeUpdateSource();
        } else if (kind === "log") {
          this.tasksLogChannel.info(`[${task.id}: ${task.info.name}] ${data}`);
        } else if (kind === "internal_error") {
          task.cancelled = true;
          log.error(`Internal error in task ${task.id}: ${data}`);
          break;
        }
      }
    } catch (e) {
      task.cancelled = true;
      log.error(`Exception raised while running task ${task.id}:`, e);
    } finally {
      this.removeTask(task);
      this.refreshView();
    }
  }
}

//////
/// Task View
//////

function formatDate(date: Date): string {
  const dateOptions = {
    day: "numeric",
    month: "long",
  };
  const timeOptions = {
    hour: "numeric",
    minute: "numeric",
  };
  // @ts-ignore
  const prettyDate = date.toLocaleDateString("en-US", dateOptions);
  // @ts-ignore
  const prettyTime = date.toLocaleTimeString("en-US", timeOptions);
  return `${prettyDate}, ${prettyTime}`;
}

// Format the duration between two dates using HH:MM:SS format
function formatDuration(startDate: Date, endDate: Date) {
  const diffInMilliseconds = endDate.getTime() - startDate.getTime();
  const diffInSeconds = Math.floor(diffInMilliseconds / 1000);
  const hours = Math.floor(diffInSeconds / 3600);
  const minutes = Math.floor((diffInSeconds % 3600) / 60);
  const seconds = diffInSeconds % 60;
  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}

// Unfortunately, we cannot disable the view loading bar:
// https://github.com/Microsoft/vscode/issues/65010
class TasksViewProvider implements vscode.TreeDataProvider<Task<any>> {
  constructor(private parent: TasksManager) {}
  private _onDidChangeTreeData = new vscode.EventEmitter<
    Task<any> | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  getTreeItem(task: Task<any>): vscode.TreeItem {
    const item = new vscode.TreeItem(task.info.name);
    // item.description = task.kind ? task.kind + " " : "";
    // item.description += `#${task.id}`;
    // item.description = task.kind ? task.kind : undefined;
    item.description = task.info.arg;
    item.tooltip = `Task #${task.id}, created on ${formatDate(task.creationDate)}`;
    // item.description = `${task.id}`;

    item.contextValue = task.contextValue();
    item.iconPath = new vscode.ThemeIcon("server-process");
    return item;
  }

  async getChildren(element?: Task<any>): Promise<Task<any>[]> {
    return this.parent.tasks;
  }

  refresh() {
    this._onDidChangeTreeData.fire();
  }
}
