### DDPG solving Openai Gym - Mountain Car Continuous problem

Without any seed it can solve within 2 episodes but on average it takes 4-6
The Learner class have a plot_Q method that will plot some very usefull graphs to tweak the model

In the training, the learner act in the environment using `0.2*action + noise` where `action` is the local actor model, `noise` is a Ornstein-Uhlenbeck process and the number `0.2` is choosen scalar to avoid the learner getting
stuck trying to use some bad actions (like +1 for all states, which can happen at first). After every epoch
the learner will be tested against the environment without any noise in the actions and if the average of the 
`n_solved` number of tests are above 90 the loop will break and the learner object will output the rewards and
steps of every episode.


## Project Instructions

1. Run the MountainCar.py file directly or import it on the notebook.

```
# This way you don't see much
> python MountainCar.py 
```

Or

```
> from MountainCar import MountainCar
> Leaner = MountainCar()
> train_hist, test_hist, solved = Learner.run_model(max_epochs=20, n_solved = 1, verbose=1)

```

2. To test the final model

```
> _, test_hist, solved = Learner.run_model(max_epochs=20, n_solved = 1, verbose=-1)
```

3. Before running make sure to have the necessary modules

