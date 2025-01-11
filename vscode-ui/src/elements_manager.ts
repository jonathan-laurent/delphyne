//////
/// Elements Manager
//////

import { DemosManager } from "./demos_manager";
import { Element } from "./elements";
import { TreeView } from "./tree_view";

// Track the liveness of elements

export class ElementsManager {
  constructor(
    private demosManager: DemosManager,
    private treeView: TreeView,
  ) {}

  registerHooks() {
    this.demosManager.onUpdate(() => this.onUpdate());
  }

  private isAlive(element: Element) {
    if (
      element.kind === "strategy_demo" ||
      element.kind === "standalone_query"
    ) {
      return this.demosManager.isAlive(element);
    }
    return false;
  }

  private onUpdate() {
    // Close the tree view when the associated element dies
    const pointedTree = this.treeView.getPointedTree();
    if (pointedTree && !this.isAlive(pointedTree.tree.origin)) {
      this.treeView.setPointedTree(null);
    }
  }
}
