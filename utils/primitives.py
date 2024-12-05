import torch

class PrimitiveGrowthFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, directions, probabilities):
        """
        Forward propagation as per the pseudo-code.
        Args:
            directions (Tensor): Potential grow directions D (N, d)
            distances (Tensor): Potential grow distances T (N,)
            probabilities (Tensor): Grow direction/distance probabilities Q (N,)

        Returns:
            Tensor: Grow direction d and grow distance t
        """
        index = torch.argmax(probabilities)  # Hard index
        index_hard = torch.nn.functional.one_hot(index, num_classes=probabilities.size(0)).float()

        grow_direction = torch.matmul(index_hard, directions)  # d = Matmul(index-hard, D)

        # Save for backward pass
        ctx.save_for_backward(directions, probabilities)

        # Return grow direction and grow distance
        return grow_direction

    @staticmethod
    def backward(ctx, grad_output_d):
        """
        Backward propagation as per the pseudo-code.
        Args:
            grad_output_d (Tensor): Gradient of grow direction d
            grad_output_t (Tensor): Gradient of grow distance t

        Returns:
            Tuple[Tensor]: Gradients w.r.t. directions, distances, probabilities
        """
        directions, probabilities = ctx.saved_tensors

        probabilities_soft = torch.softmax(probabilities, dim=0)

        grad_directions = None  # No gradient for directions

        grad_probabilities = torch.sum(directions * grad_output_d, dim=1) * probabilities_soft
        return grad_directions, grad_probabilities

# Usage
# directions = torch.randn(5, 3, requires_grad=False)  # 5 directions, each in 3D space
# prob_paremeters = torch.randn(5, requires_grad=True)  # Probabilities for directions
# # probabilities = torch.softmax(prob_paremeters, dim=0)  # Ensure they sum up to 1

# # Apply the custom function
# grow_direction = PrimitiveGrowthFunction.apply(directions, prob_paremeters)

# # Compute gradients
# grow_direction.sum().backward()  # Example loss
# print(f"Gradients w.r.t. directions: {directions.grad}")
# print(f"Gradients w.r.t. probabilities: {prob_paremeters.grad}")
# # print(f"Gradienrs w.r.t. grow direction: {prob_paremeters.grad}")
