import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "BiRefNet.BiRefNetNode",
  async nodeCreated(node) {
     console.log("created BiRefNet_Node", node);
    // if (node.comfyClass !== "BiRefNet_Node") {
    //   return;
    // } else {
    //   console.log("created BiRefNet_Node", node);
    // }
  },
});
