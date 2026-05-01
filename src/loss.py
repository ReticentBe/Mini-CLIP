import torch
from torch import nn
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    '''
    A custom PyTorch 'autograd.Function' to implement a differentiable 'all_gather'
    'dist.all_gather' breaks the computation graph and does not backpropagate gradients 
    across distributed workers. This layer is desiged for gradients to correctly accumulate
    and route back to the appropriate gpus during distributed contrastive learning
    '''
    @staticmethod
    def forward(ctx, x):
        # Initialize a list to receive data from all gpus
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)
    
    @staticmethod
    def backward(ctx, *grads):
        '''
        Accumulates cross-GPU gradients and routes them to the local model
        grads: A tuple of gradients, one for each rank's output. Stack them into a
        single tensor of shape [world_size, batch_size, dim]
        '''
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)   # Sum the gradients across all GPUs
        return all_gradients[dist.get_rank()]   
        # Extract and return the gradient corresponding to the current local rank
    
def all_gather_with_grad(tensors):
    """
    Gather tensors from all processes and concatenates them along the batch dimension
    while preserving the computation graph for backpropagation
    Input tensors: Local tensor of shape [batch_size, dim]
    Returns: Global concatenated tensor of shape [batch_size * world_size, dim]
    """
    gathered = GatherLayer.apply(tensors)
    return torch.cat(gathered, dim=0)



'''
Loss class design without DDP

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits_image, logits_text):
        """
        logits_image: [Batch_size, Batch_size]
        logits_text: [Batch_size, Batch_size]
        """
        batch_size = logits_image.shape[0]
        device = logits_image.device

        targets= torch.arange(batch_size, device=device)

        loss_i = self.criterion(logits_image, targets)

        loss_t = self.criterion(logits_text, targets)

        total_loss = (loss_i + loss_t)/2
        
        return total_loss
'''

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_embeddings, text_embeddings, logit_scale):
        rank = dist.get_rank()
        batch_size = image_embeddings.shape[0]

        global_image_embeddings = all_gather_with_grad(image_embeddings)
        global_text_embeddings = all_gather_with_grad(text_embeddings)

        logits_per_image = logit_scale * image_embeddings @ global_text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ global_image_embeddings.T
        # [B, D] @ [D, B * world_size] -> [B, B * world_size]

        labels = torch.arange(batch_size, device=image_embeddings.device) + rank * batch_size

        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)

        return (loss_i + loss_t) / 2




        