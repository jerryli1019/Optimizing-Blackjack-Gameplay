To run the blackjack program, do:
```
python main.py
```
To run deterministic tester for MC algorithm:
```
python main.py -t 1 -a 1
```
To run convergence tester for all algorithms:
```
python main.py -t 2
```
Once in-game, the following keyboard options are available:
- 'h': hit
- 's': stand
- 'm': toggle MC learning
- 't': toggle TD learning
- 'q': toggle Q-learning
- 'a': toggle autoplay
- '1': save the AI state (not the game state)
- '2': load from saved AI state