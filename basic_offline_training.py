import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from helpers import DiabetesDataset, device, compute_reward_torch
from tqdm import tqdm
import csv
from torch.utils.tensorboard import SummaryWriter

class SimpleSAC(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor - simple deterministic network
        self.actor = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        # Critic - single Q-network
        self.critic = nn.Sequential(
            nn.Linear(10, 64),  # state + action
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Target network for stability
        self.critic_target = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Disable gradient tracking for target network
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def act(self, state):
        return self.actor(state)

def train_basic(dataset_path, csv_file='basic_training_stats.csv', epochs=100, 
                lr=3e-4, batch_size=256, print_interval=10):
    # Setup data
    dataset = DiabetesDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer
    model = SimpleSAC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Fixed hyperparameters
    gamma = 0.99  # Discount factor
    tau = 0.01    # Target network update rate
    bc_weight = 0.1  # Behavioral cloning weight
    
    # Logging
    writer = SummaryWriter()
    
    # Initialize CSV
    with open(csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch', 'Iteration', 'TD Loss', 'BC Loss', 
                           'Total Loss', 'Q Value', 'Action_Mean', 'Action_Std'])
    
    # Training loop
    global_step = 0
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_td_loss = 0
        epoch_bc_loss = 0
        epoch_total_loss = 0
        epoch_q_value = 0
        epoch_action_mean = 0
        epoch_action_std = 0
        batch_count = 0
        
        for i, batch in enumerate(loader):
            # Transfer to device
            states = batch["state"].to(device)
            actions = batch["action"].to(device)
            rewards = batch["reward"].to(device).unsqueeze(1)
            next_states = batch["next_state"].to(device)
            dones = batch["done"].to(device).unsqueeze(1)
            
            # Critic Update
            with torch.no_grad():
                next_actions = model.actor(next_states)
                target_q = rewards + (1 - dones) * gamma * model.critic_target(
                    torch.cat([next_states, next_actions], 1))
            
            current_q = model.critic(torch.cat([states, actions], 1))
            td_loss = F.mse_loss(current_q, target_q)
            
            # Behavioral Cloning Loss
            pred_actions = model.actor(states)
            bc_loss = F.mse_loss(pred_actions, actions)
            
            # Total Loss
            total_loss = td_loss + bc_weight * bc_loss
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Target Network Update
            with torch.no_grad():
                for t_param, param in zip(model.critic_target.parameters(), model.critic.parameters()):
                    t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
            
            # Metrics
            epoch_td_loss += td_loss.item()
            epoch_bc_loss += bc_loss.item()
            epoch_total_loss += total_loss.item()
            epoch_q_value += current_q.mean().item()
            epoch_action_mean += pred_actions.mean().item()
            epoch_action_std += pred_actions.std().item()
            batch_count += 1
            global_step += 1
            
            # Logging
            if i % print_interval == 0 and batch_count > 0:
                avg_td_loss = epoch_td_loss / batch_count
                avg_bc_loss = epoch_bc_loss / batch_count
                avg_total_loss = epoch_total_loss / batch_count
                avg_q_value = epoch_q_value / batch_count
                avg_action_mean = epoch_action_mean / batch_count
                avg_action_std = epoch_action_std / batch_count
                
                # TensorBoard
                writer.add_scalar('Loss/TD', avg_td_loss, global_step)
                writer.add_scalar('Loss/BC', avg_bc_loss, global_step)
                writer.add_scalar('Loss/Total', avg_total_loss, global_step)
                writer.add_scalar('Values/Q', avg_q_value, global_step)
                writer.add_scalar('Actions/Mean', avg_action_mean, global_step)
                writer.add_scalar('Actions/Std', avg_action_std, global_step)
                
                # CSV
                with open(csv_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        epoch, i, avg_td_loss, avg_bc_loss, avg_total_loss,
                        avg_q_value, avg_action_mean, avg_action_std
                    ])
                
                # Reset metrics
                epoch_td_loss = 0
                epoch_bc_loss = 0
                epoch_total_loss = 0
                epoch_q_value = 0
                epoch_action_mean = 0
                epoch_action_std = 0
                batch_count = 0
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"basic_sac_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), "basic_sac_final.pth")
    writer.close()
    
    return model

if __name__ == "__main__":
    train_basic(
        dataset_path="datasets/processed/563-train.csv",
        epochs=100,
        lr=3e-4,
        batch_size=256
    )
