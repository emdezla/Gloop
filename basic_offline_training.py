import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from helpers import DiabetesDataset, device, compute_reward_torch
from tqdm import tqdm
import csv
from torch.utils.tensorboard import SummaryWriter

# Custom Mish activation for improved training stability
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SimpleSAC(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor - simple deterministic network with improved capacity
        self.actor = nn.Sequential(
            nn.Linear(8, 128),
            nn.LayerNorm(128),
            Mish(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            Mish(),
            nn.Linear(128, 2),
            nn.Tanh()
        )
        
        # Twin Critics - to reduce overestimation bias
        self.critic1 = nn.Sequential(
            nn.Linear(10, 128),  # Reduced from 256
            Mish(),
            nn.Linear(128, 128),
            Mish(),
            nn.Linear(128, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(10, 128),  # Reduced from 256
            Mish(),
            nn.Linear(128, 128),
            Mish(),
            nn.Linear(128, 1)
        )
        
        # Target networks for stability
        self.critic1_target = nn.Sequential(
            nn.Linear(10, 128),
            Mish(),
            nn.Linear(128, 128),
            Mish(),
            nn.Linear(128, 1)
        )
        
        self.critic2_target = nn.Sequential(
            nn.Linear(10, 128),
            Mish(),
            nn.Linear(128, 128),
            Mish(),
            nn.Linear(128, 1)
        )
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Disable gradient tracking for target networks
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

    def act(self, state, noise_scale=0.1):
        action = self.actor(state)
        if self.training:  # Add noise only during training
            noise = torch.randn_like(action) * noise_scale
            return torch.clamp(action + noise, -1, 1)
        return action

def train_basic(dataset_path, csv_file='basic_training_stats.csv', epochs=500, 
                lr=3e-5, batch_size=256, print_interval=10):
    # Setup data
    dataset = DiabetesDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and separate optimizers
    model = SimpleSAC().to(device)
    actor_optim = optim.Adam(model.actor.parameters(), lr=lr)
    critic_optim = optim.Adam(
        list(model.critic1.parameters()) + list(model.critic2.parameters()), 
        lr=lr*2  # Higher learning rate for critics
    )
    
    # Fixed hyperparameters
    gamma = 0.99  # Discount factor - increased
    tau = 0.005   # Target network update rate - slower updates
    bc_weight = 0.3  # Behavioral cloning weight - increased regularization
    grad_penalty_weight = 10.0  # Weight for gradient penalty - increased
    noise_scale = 0.05  # Reduced action noise
    
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
                next_actions = model.act(next_states, noise_scale=noise_scale)
                target_q1 = model.critic1_target(torch.cat([next_states, next_actions], 1))
                target_q2 = model.critic2_target(torch.cat([next_states, next_actions], 1))
                # Use minimum of two Q-values to reduce overestimation bias
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - dones) * gamma * target_q
                target_q = torch.clamp(target_q, -10, 10)  # Add clipping to prevent explosion
            
            # Current Q-values from both critics
            current_q1 = model.critic1(torch.cat([states, actions], 1))
            current_q2 = model.critic2(torch.cat([states, actions], 1))
            
            # TD losses for both critics
            td_loss1 = F.mse_loss(current_q1, target_q)
            td_loss2 = F.mse_loss(current_q2, target_q)
            td_loss = td_loss1 + td_loss2
            
            # Improved gradient penalty (WGAN-style)
            alpha = 1.0
            random_alpha = torch.rand(states.size(0), 1, device=device)
            interpolated = (random_alpha * states + (1 - random_alpha) * next_states).requires_grad_(True)
            critic_interpolated = model.critic1(torch.cat([interpolated, actions], 1))
            gradients = torch.autograd.grad(
                outputs=critic_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones_like(critic_interpolated),
                create_graph=True,
                retain_graph=True
            )[0]
            grad_penalty = grad_penalty_weight * ((gradients.norm(2, dim=1) - 1.0).pow(2)).mean()
            td_loss += grad_penalty
            
            # Behavioral Cloning Loss with annealed weighting
            pred_actions = model.act(states, noise_scale=0.0)  # No noise for BC loss
            bc_loss = F.mse_loss(pred_actions, actions)
            
            # Annealed BC weight that decreases over time
            effective_bc_weight = bc_weight * (1 - epoch/epochs)  # Reverse annealing
            
            # Total Loss
            total_loss = td_loss + effective_bc_weight * bc_loss
            
            # Separate optimization steps for critics and actor
            critic_optim.zero_grad()
            td_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.critic1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.critic2.parameters(), 1.0)
            critic_optim.step()
            
            actor_optim.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
            actor_optim.step()
            
            # Target Network Update
            with torch.no_grad():
                for t_param, param in zip(model.critic1_target.parameters(), model.critic1.parameters()):
                    t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
                for t_param, param in zip(model.critic2_target.parameters(), model.critic2.parameters()):
                    t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
            
            # Metrics
            epoch_td_loss += td_loss.item()
            epoch_bc_loss += bc_loss.item()
            epoch_total_loss += total_loss.item()
            epoch_q_value += (current_q1.mean().item() + current_q2.mean().item()) / 2
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
        epochs=500,
        lr=1e-4,
        batch_size=256
    )
