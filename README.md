# CorrectPose2


1. user opens the app/website
2. user selects the yoga he wants to do
3. camera starts recording
4. body points get detected:
    using mediapipe. then angles get calculated. those angles are then passed onto the model as inputs.
5. a trained model tells whether the user is performing correctly or not.
6. if the user performs correctly for the recommended time bracketts, he gets a green light.
    else: he he gets feedback to improve the position until he gets it right.
7. user can go back and select a different yoga pose
8. repeat.


In order to achieve this, we have to create seperate models for all the yoga poses and exercises in our catalogue.
Each model should be a binary classification model telling whether the position is correct or not. 