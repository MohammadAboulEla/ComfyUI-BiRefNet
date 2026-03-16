import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const showToast = (severity, summary, detail) => {
  app.extensionManager.toast.add({ severity, summary, detail, life: 8000 });
};

const checkModels = async () => {
  try {
    const models = await api.getModels("BiRefNet");
    if (!models || models.length === 0) {
      showToast(
        "info",
        "Downloading Model...",
        "First-time setup: BiRefNet model is being installed to your models directory.",
      );
    }
  } catch (error) {
    console.error("Error fetching model:", error);
  }
};

app.registerExtension({
  name: "BiRefNet.BiRefNetNode",
  async nodeCreated(node) {
    if (node.comfyClass !== "BiRefNet_Node") {
      return;
    }
    node.onExecutionStart = () => {
      checkModels();
    };
  },
});
