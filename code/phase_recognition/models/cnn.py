import torchvision.models as models
import torch.nn as nn


class ResNet50Model(nn.Module):
    def __init__(self, hparams):
        super(ResNet50Model, self).__init__()
        self.model = models.resnet50(pretrained=hparams.pretrained)
        self.model.fc = Identity()  # type: ignore[assignment]
        self.fc_phase = nn.Linear(2048, hparams.out_features)

    def forward(self, x):
        stem = self.model(x)
        phase = self.fc_phase(stem)
        return stem, phase

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        group = parser.add_argument_group(title='ResNet50Model specific args')
        group.add_argument("--pretrained", action="store_true",
                           help="pretrained on ImageNet")
        group.add_argument("--model_specific_batch_size_max", type=int, default=80)
        return parser


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
