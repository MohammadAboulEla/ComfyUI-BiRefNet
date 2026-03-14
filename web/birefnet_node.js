import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "BiRefNet.BiRefNetNode",
  async nodeCreated(node) {
    if (node.comfyClass !== "BiRefNet_Node") {
      return;
    }
    // frontend future
    // console.log("created BiRefNet_Node", node);
  },
});
