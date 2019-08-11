
from rl.pg.agents import VPGAgent
from rl.pg.policies import MlpPolicy

# create environment
env = gym.make( 'CartPole-v0' )

# create a VPG agent (already configured)
agent = VPGAgent( policy = MlpPolicy() )

# perhaps load pre-trained agent
agent.load( 'vpg_cartpole' )

for _ in range( 1000 ) :
    s = env.reset()

    while True :
        a = agent.act( s )
        s_next, r, done, _ = env.step()
        agent.update( ( s, a, r, s_next, done ) )

        s = s_next

        if done :
            break