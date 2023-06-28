from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--data_path", type=str, help="Path to data")
    parser.add_argument("--data_asset_name", type=str, help="name of data asset you want to give")
    
    args = parser.parse_args()

    return args


def register_data_asset(args):
# registering data-asset

    my_data = Data(
        path = args.data_path,
        type = AssetTypes.URI_FILE,
        description = "Data asset pointing to a local file, automatically uploaded to the default datastore",
        name = args.data_asset_name
    )

    ws.data.create_or_update(my_data)

    
    
if __name__ == "__main__":

    args = parse_args()
