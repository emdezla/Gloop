from helpers import *

dataset  = DiabetesDataset(csv_file="datasets/processed/563-train.csv")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = SACCQL().to(device)
optimizer_actor = optim.Adam(model.actor.parameters(), lr=3e-4)
optimizer_critic = optim.Adam(list(model.q1.parameters()) + list(model.q2.parameters()), lr=3e-4)

torch.autograd.set_detect_anomaly(True)
print_interval = 100
writer = SummaryWriter()
csv_file = 'training_stats1.csv'


with open(csv_file, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Epoch', 'Iteration', 'TD Loss', 'CQL Penalty', 
                        'Critic Loss', 'Actor Loss', 'Q1 Value', 'Q2 Value',
                        'Action_Mean', 'Action_Std', 'Entropy'])

# Training loop
for epoch in tqdm(range(1000), desc="Training Progress"):
    metrics = {
        'td': 0.0, 'cql': 0.0, 'critic': 0.0, 'actor': 0.0,
        'q1': 0.0, 'q2': 0.0, 'action_mean': 0.0, 'action_std': 0.0,
        'entropy': 0.0, 'count': 0
    }

    for i, batch in enumerate(dataloader):
        # --- Data Preparation ---
        states = batch["state"].to(device)
        dataset_actions = batch["action"].to(device)
        rewards = batch["reward"].to(device).unsqueeze(1)
        next_states = batch["next_state"].to(device)
        dones = batch["done"].to(device).unsqueeze(1)

        # --- Critic Update ---
        with torch.no_grad():
            # Get policy actions for next states
            next_mean, next_log_std = model(next_states)
            next_std = next_log_std.exp()
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_actions = torch.tanh(next_normal.rsample()) * model.action_scale
            
            # Target Q calculation
            q1_next = model.q1_target(torch.cat([next_states, next_actions], 1))
            q2_next = model.q2_target(torch.cat([next_states, next_actions], 1))
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + (1 - dones) * 0.99 * q_next

        # Current Q estimates
        current_q1 = model.q1(torch.cat([states, dataset_actions], 1))
        current_q2 = model.q2(torch.cat([states, dataset_actions], 1))
        
        # TD Loss
        td_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # CQL Penalty
        cql_penalty = compute_cql_penalty(states, dataset_actions, model)
        critic_loss = td_loss + cql_weight * cql_penalty

        # Critic optimization
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        # --- Actor Update ---
        # Generate actions from current policy
        mean, log_std = model(states) 
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)  # Squashed to [-1, 1]
        pred_actions = y_t * model.action_scale  # Scaled to insulin range
        
        # Calculate log probs with tanh correction
        log_probs = normal.log_prob(x_t).sum(1)
        log_probs -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1)
        entropy = -log_probs.mean()

        # Q-values for policy actions
        q1_pred = model.q1(torch.cat([states, pred_actions], 1))
        q2_pred = model.q2(torch.cat([states, pred_actions], 1))
        
        # Actor loss with entropy regularization
        actor_loss = -torch.min(q1_pred, q2_pred).mean() + alpha * entropy

        # Actor optimization
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        # --- Target Network Update ---
        model.update_targets()

        # --- Metrics Collection ---
        metrics['td'] += td_loss.item()
        metrics['cql'] += cql_penalty.item()
        metrics['critic'] += critic_loss.item()
        metrics['actor'] += actor_loss.item()
        metrics['q1'] += q1_pred.mean().item()
        metrics['q2'] += q2_pred.mean().item()
        metrics['action_mean'] += pred_actions.mean().item()
        metrics['action_std'] += pred_actions.std().item()
        metrics['entropy'] += entropy.item()
        metrics['count'] += 1

        # --- Logging ---
        if metrics['count'] > 0:
            avg_metrics = {k: v/metrics['count'] for k, v in metrics.items() if k != 'count'}
            
            # TensorBoard Logging
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Loss/TD', avg_metrics['td'], global_step)
            writer.add_scalar('Loss/CQL', avg_metrics['cql'], global_step)
            writer.add_scalar('Loss/Critic', avg_metrics['critic'], global_step)
            writer.add_scalar('Loss/Actor', avg_metrics['actor'], global_step)
            writer.add_scalar('Q_Values/Q1', avg_metrics['q1'], global_step)
            writer.add_scalar('Q_Values/Q2', avg_metrics['q2'], global_step)
            writer.add_scalar('Actions/Mean', avg_metrics['action_mean'], global_step)
            writer.add_scalar('Actions/Std', avg_metrics['action_std'], global_step)
            writer.add_scalar('Entropy', avg_metrics['entropy'], global_step)

            # CSV Logging
            with open(csv_file, 'a', newline='') as f:
                csv_writer.writerow([
                    epoch, i,
                    avg_metrics['td'], avg_metrics['cql'],
                    avg_metrics['critic'], avg_metrics['actor'],
                    avg_metrics['q1'], avg_metrics['q2'],
                    avg_metrics['action_mean'], avg_metrics['action_std'],
                    avg_metrics['entropy']
                ])
            
            # Reset metrics
            metrics = {k: 0.0 for k in metrics}
            metrics['count'] = 0