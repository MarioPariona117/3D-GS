import torch

class PrimitiveGrowthFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, directions, probabilities):
        """
        Forward propagation as per the pseudo-code.
        Args:
            directions (Tensor): Potential grow directions D (N, d)
            probabilities (Tensor): Grow direction probabilities Q (N,)
        Returns:
            Tensor: Grow direction d
        """
        index = torch.argmax(probabilities, dim=-1)  # Hard index
        # print(index, index.shape, "argmax")
        # print(probabilities, probabilities.shape)
        # print(probabilities.size(1))
        index_hard = torch.nn.functional.one_hot(index, num_classes=probabilities.size(1)).float()
        # print(index_hard, index_hard.shape)
        grow_direction = torch.matmul(index_hard.unsqueeze(1), directions).squeeze(1) # [1, 3]
        # print(grow_direction)

        # Save for backward pass
        ctx.save_for_backward(directions, probabilities)

        # Return grow direction
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
        
        # print(directions, directions.shape, "directions") 
        # print(probabilities, probabilities.shape, "probabilities")
        # print(grad_output_d, grad_output_d.shape, "grad_output_d")
        
        # Step 1: Compute index-soft using softmax
        index_soft = torch.softmax(probabilities, dim=-1)
        # print(index_soft, index_soft.shape, "index_soft")
        
        # Step 2: Compute gradients
        grad_directions = torch.matmul(index_soft.unsqueeze(2), grad_output_d.unsqueeze(1)).squeeze(0)
        grad_probabilities = torch.sum(directions * grad_output_d.unsqueeze(1), dim=-1) * index_soft
        
        return grad_directions, grad_probabilities

# # Usage
# directions = torch.randn(2, 5, 3, requires_grad=False)  # 5 directions, each in 3D space
# # print(directions.shape)
# # directions = directions.unsqueeze(0)
# print(directions.shape)
# prob_paremeters = torch.randn(2,5, requires_grad=True)  # Probabilities for directions
# # probabilities = torch.softmax(prob_paremeters, dim=0)  # Ensure they sum up to 1
# # print(prob_paremeters.shape)
# # prob_paremeters = prob_paremeters.unsqueeze(0)
# print(prob_paremeters.shape)
# # Apply the custom function
# grow_direction = PrimitiveGrowthFunction.apply(directions, prob_paremeters)

# # # Compute gradients
# grow_direction.sum().backward()  # Example loss
# print(f"Gradients w.r.t. directions: {directions.grad}")
# print(f"Gradients w.r.t. probabilities: {prob_paremeters.grad}")
# # # print(f"Gradienrs w.r.t. grow direction: {prob_paremeters.grad}")
