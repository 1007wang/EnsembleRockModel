import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx):
        """Generate Grad-CAM heatmap"""
        # Save original model state
        was_training = self.model.training
        self.model.eval()  # Set to evaluation mode

        # Ensure input requires gradients
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        try:
            # Forward pass
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

            if output.dim() == 1:
                output = output.unsqueeze(0)

            # Zero gradients
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            self.model.zero_grad()

            # Create one-hot encoding
            one_hot = torch.zeros_like(output)
            one_hot[0][class_idx] = 1

            # Backward pass
            output.backward(gradient=one_hot, retain_graph=True)

            # Ensure gradients and activations exist
            if self.gradients is None or self.activations is None:
                raise ValueError("Failed to obtain gradients or activations")

            # Calculate weights
            weights = torch.mean(self.gradients, dim=(2, 3))[0]
            cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(input_tensor.device)

            # Generate CAM
            for i, w in enumerate(weights):
                cam += w * self.activations[0, i]

            cam = torch.relu(cam)
            cam = cam.cpu().detach().numpy()
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-7)
            cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))

        finally:
            # Restore original model state
            self.model.train(was_training)

        return cam

    def overlay_cam(self, image, cam, alpha=0.5):
        """Overlay CAM onto original image"""
        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        
        # Overlay image
        overlayed_image = heatmap * alpha + image * (1 - alpha)
        overlayed_image = overlayed_image / np.max(overlayed_image)
        
        return overlayed_image 