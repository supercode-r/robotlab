import torch
import numpy as np
import torch.nn as nn


class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        self.num_class = 9

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, stride=2),
        )


        # self.fc2 = nn.Sequential(
        #     nn.Linear(1024, 128),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.BatchNorm1d(128),
        #     nn.Dropout(0.5)
        # )

        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, self.num_class, kernel_size=1, stride=1, padding=0),
        )
        self.softmax = nn.Softmax()

        ### Second Try

        # self.conv1 = torch.nn.conv2d(1,2,3, stride = 1)
        # self.relu1 = torch.nn.functional.leaky_relu(negative_slope = 0.2) #94x94
        #
        # self.conv2 = torch.nn.functional.conv2d(2,4,3, stride = 2)
        # self.relu2 = torch.nn.functional.leaky_relu(negative_slope = 0.2)
        #
        # self.conv3 = torch.nn.functional.conv2d(4,8,3, stride = 2)
        # self.relu3 = torch.nn.functional.leaky_relu(negative_slope = 0.2)
        #
        # self.fc1 = torch.nn.functional.linear(8*22*22, 64)
        # self.relu4 = torch.nn.functional.leaky_relu(negative_slope = 0.2)
        #
        # self.fc2 = torch.nn.functional.linear(64,32)
        # self.relu5 = torch.nn.functional.leaky_relu(negative_slope=0.2)
        #
        # self.fc3 = torch.nn.functional.linear(32,10)
        # self.relu6 = torch.nn.functional.leaky_relu(negative_slope=0.2)

        #### Third try

        # layers = [
        #     mnn.layers.Input(input_shape=[96, 96, n_channels]),
        #     mnn.layers.Conv2d(filters=16, kernel_size=5, stride=4),
        #     mnn.layers.ReLU(),
        #     mnn.layers.Dropout(drop_probability=0.5),
        #     mnn.layers.Conv2d(filters=32, kernel_size=3, stride=2),
        #     mnn.layers.ReLU(),
        #     mnn.layers.Dropout(drop_probability=0.5),
        #     mnn.layers.Flatten(),
        #     mnn.layers.Linear(n_units=128),
        #     mnn.layers.Linear(n_units=utils.n_actions),
        # ]
        # model = mnn.models.Classifier_From_Layers(layers)
        # gpu = torch.device('cuda')

    def forward(self, observation):
        observation = self.layer1(observation)
        observation = self.layer2(observation)
        observation = self.layer3(observation)
        observation = self.layer4(observation)
        # observation = observation.view(observation.size(0), -1)
        # y = y.view(y.size(0), -1)
        # xy = torch.cat((x, y), dim=1)
        # observation = self.fc1(observation)
        # observation = self.fc2(observation)
        observation = self.out(observation)
        observation = torch.flatten(observation)
        observation = self.softmax(observation)

        ### second try

        # labels = observation[1].data
        # scores = observation[0].data
        # # Normalizing to avoid instability
        # scores -= np.max(scores, axis=1, keepdims=True)
        # # Compute Softmax activations
        # exp_scores = np.exp(scores)
        # probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # logprobs = np.zeros([observation[0].num, 1])
        # # Compute cross-entropy loss
        # for r in range(observation[0].num):  # For each element in the batch
        #     scale_factor = 1 / float(np.count_nonzero(labels[r, :]))
        #     for c in range(len(labels[r, :])):  # For each class
        #         if labels[r, c] != 0:  # Positive classes
        #             logprobs[r] += -np.log(probs[r, c]) * labels[
        #                 r, c] * scale_factor  # We sum the loss per class for each element of the batch
        #
        # data_loss = np.sum(logprobs) / bottom[0].num
        #
        # self.diff[...] = probs  # Store softmax activations
        # top[0].data[...] = data_loss  # Store loss
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        return observation

    def actions_to_classes(self, actions):
        self.action_sets = list(set(tuple(action.tolist()) for action in actions))
        self.classes = np.eye(len(self.action_sets)).tolist()
        action_to_classes = []
        for action in actions:
            action_to_classes.append(torch.tensor(self.classes[self.action_sets.index(tuple(action))]))

        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        # steer = set(action[0].item() for action in actions)
        # gas = set(action[1].item() for action in actions)
        # brake = set(action[2].item() for action in actions)
        # for action in actions:
        #     steer = action[0].item()
        #     gas = action[1].item()
        #     brake = action[2].item()
        #     if steer < 0 and gas == 0 and brake == 0:
        #         actionclass = classes[0]
        #     elif steer < 0 and gas > 0 and brake == 0:
        #         actionclass = classes[1]
        #     elif steer < 0 and gas == 0 and brake > 0:
        #         actionclass = classes[2]
        #     elif steer == 0 and gas == 0 and brake == 0:
        #         actionclass = classes[3]
        #     elif steer == 0 and gas > 0 and brake == 0:
        #         actionclass = classes[4]
        #     elif steer == 0 and gas == 0 and brake > 0:
        #         actionclass = classes[5]
        #     elif steer > 0 and gas == 0 and brake == 0:
        #         actionclass = classes[6]
        #     elif steer > 0 and gas > 0 and brake == 0:
        #         actionclass = classes[7]
        #     elif steer > 0 and gas == 0 and brake > 0:
        #         actionclass = classes[8]
        #     else:
        #         actionclass = None
        #         print("class not defined")
        #
        #     action_to_class.append(torch.tensor(actionclass, dtype=int))

        return action_to_classes

    def scores_to_action(self, scores):

        scores_to_actions = []
        for score in scores:
            scores_to_actions.append(self.action_sets[self.classes.index(score.tolist())])
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        return scores_to_actions

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
