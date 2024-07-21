import hydra
from model import train, log_metadata
from data import read_features
from omegaconf import OmegaConf
import mlflow 


def run(cfg):
    train_data_version = str(cfg.train_data_version)
    test_data_version = str(cfg.test_data_version)

    X_train, y_train = read_features(name="features_target",
                                     version=train_data_version,
                                     size=1.0)
    X_test, y_test = read_features(name="features_target",
                                   version=test_data_version,
                                   size=0.5)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    gs = train(X_train, y_train, cfg=cfg)

    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    # print(OmegaConf.to_yaml(cfg))
    # print(cfg.data.target_cols[0])
    run(cfg)


if __name__ == '__main__':
    main()
