model_pi_theta.train()

model_pi_theta_old.eval()

model_pi_theta_ref.eval()

for iteration 
    for (batch)
        with torch.no_grad: 
            samples = model_pi_theta.sample()
        rewards = advantage_fun(samples)
        p_old = model_pi_theta_old(samples)
        p_ref = model_pi_theta_ref(samples)
        while (GRPO_steps):
            p_theta = model_pi_theta(samples)
            loss = p_theta/p_old * reward -KL(p_theta,p_ref )
            loss.backward()
        model_pi_theta_old = model_pi_theta
    model_pi_theta_ref = model_pi_theta
