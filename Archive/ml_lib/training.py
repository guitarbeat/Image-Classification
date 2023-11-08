# training.py

from tensorflow.keras import backend as K
import gc
import os
import pandas as pd

class ExcelMetricsSaver(Callback):
    def __init__(self, writer, sheet_name):
        super().__init__()
        self.writer = writer
        self.sheet_name = sheet_name

    def on_epoch_end(self, epoch, logs=None):
        self._save_metrics_to_excel(epoch, logs or {})

    def _save_metrics_to_excel(self, epoch, logs):
        logs['epoch'] = epoch + 1
        epoch_df = pd.DataFrame([logs])

        try:
            if epoch == 0:
                epoch_df.to_excel(self.writer, sheet_name=self.sheet_name, index=False)
            else:
                writer_sheets = self.writer.sheets
                start_row = writer_sheets[self.sheet_name].max_row
                epoch_df.to_excel(self.writer, sheet_name=self.sheet_name, startrow=start_row, header=False, index=False)
        except Exception as e:
            print(f"Error while saving metrics: {e}")


def handle_memory_after_training(model):
    K.clear_session()
    del model
    gc.collect()


def train_and_save_metrics(train_dataset, valid_dataset, test_dataset, compiled_models, prepared_class_weights):
    if not all([train_dataset, valid_dataset, test_dataset]):
        print("One or more datasets are None. Exiting.")
        return

    excel_filename = f"{config['Experiment']['NAME']}.xlsx"
    excel_path = os.path.join("./", excel_filename)

    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        pd.DataFrame().to_excel(writer, sheet_name="InitializationSheet")

        for model_name, model_info in compiled_models.items():
            if not _process_single_model(writer, model_name, model_info, train_dataset, valid_dataset, prepared_class_weights):
                continue
            _autosave_excel(writer, excel_path)

        print(f"\nSaved all metrics to {excel_path}")


def _process_single_model(writer, model_name, model_info, train_dataset, valid_dataset, prepared_class_weights):
    model = model_info.get('model')
    callbacks = model_info.get('callbacks')

    if model is None or callbacks is None:
        print(f"Model or callbacks for {model_name} are None. Skipping.")
        return False

    print(f"Training {model_name} for multi-output...")

    excel_saver = ExcelMetricsSaver(writer, sheet_name=model_name)
    callbacks.append(excel_saver)

    model_output_names = [layer.name for layer in model.layers if 'output' in layer.name]
    current_class_weights = {name: prepared_class_weights[name] for name in model_output_names}

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=config['Model']['EPOCHS'],
        callbacks=callbacks,
        class_weight=current_class_weights
    )

    handle_memory_after_training(model)

    print(f"\nTraining for {model_name} completed.")
    return True


def _autosave_excel(writer, excel_path):
    temp_excel_path = excel_path.replace('.xlsx', '_temp.xlsx')
    writer.save(temp_excel_path)
    if os.path.exists(excel_path):
        backup_path = excel_path.replace('.xlsx', '_backup.xlsx')
        os.replace(excel_path, backup_path)
    os.replace(temp_excel_path, excel_path)


# Assuming you've already defined and initialized all your datasets and configs
train_and_save_metrics(train_dataset, valid_dataset, test_dataset, compiled_models, prepared_class_weights)


