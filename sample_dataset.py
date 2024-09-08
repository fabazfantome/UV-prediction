from torch.utils.data import Dataset
from torch import nn, Tensor

class SampleDataset(Dataset):
    def __init__(self) -> None:
        ## Defini todas los atributos necesarios. Ejemplo: ## self.image_paths (contiene el path hacia todas las imagenes)
    def __len__(self) -> int:
        ## Defini la forma en que se calcula la cantidad de samples en tu dataset, por ejemplo: return len(self.paraguay_uv_dataframe)

    def __getitem__(self, idx: int | slice) -> Tuple[Tensor, Tensor]:
        # Defini la forma en la que vas a distribuir los datos, tienen que ser a tensores siempre, entonces tene en cuenta eso
