
import torch


def run_one_epoch(model, dataloader, *, criterion=None, optimizer=None, device=None):
    r'''Run one epoch of training or testing.

    If `criterion` and `optimizer` are both provided ==> This is training
    If `criterion` is provided, but `optimizer` is not ==> This is validation/testing
    If `criterion` is not provided ==> This is inference, and the dataloader might be just a list of inputs
    '''
    model.train() if optimizer else model.eval()
    epoch_loss = 0.0
    num_correct = 0
    total = 0
    all_outputs = None

    with torch.set_grad_enabled(optimizer is not None):
        for batch_idx, batch in enumerate(dataloader):
            if criterion:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs = batch
                inputs = inputs.to(device)

            outputs = model(inputs)

            if criterion:
                # Training or validation
                loss = criterion(outputs, targets)
                if optimizer:
                    # Training
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                loss = loss.item()
                epoch_loss += loss * inputs.size(0)
                _, predicted = outputs.max(1)
                total += inputs.size(0)
                num_correct += predicted.eq(targets).sum().item()
            else:
                # Accumulate the outputs
                if batch_idx == 0:
                    all_outputs = outputs
                else:
                    all_outputs = torch.cat((all_outputs, outputs), dim=0)
    if criterion:
        return epoch_loss / total, num_correct / total
    else:
        return all_outputs
