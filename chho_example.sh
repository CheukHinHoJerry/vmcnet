vmc-molecule \
    --config.problem.ion_pos="((0.0, 0.0, -2.0), (0.0, 0.0, 2.0))" \
    --config.problem.ion_charges="(7.0, 7.0)" \
    --config.problem.nelec="(7, 7)" \
    --config.vmc.nburn=1 \
    --config.vmc.optimizer_type="warmsr" \
    --config.vmc.optimizer.warmsr.srft_rank=300 \
    --config.vmc.optimizer.warmsr.mu=0.99 \
    --config.vmc.optimizer.warmsr.damping=0.001 \
    --config.vmc.nchains=500 \