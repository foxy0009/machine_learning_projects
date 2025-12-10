import json

notebook_path = 'Model_training_testing.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "export_artifacts",
    "metadata": {},
    "outputs": [],
    "source": [
        "import joblib\n",
        "\n",
        "# Save model columns\n",
        "model_columns = list(X.columns)\n",
        "joblib.dump(model_columns, 'model_columns.pkl')\n",
        "print(f\"Saved model_columns.pkl with {len(model_columns)} columns\")\n",
        "\n",
        "# Save scaler\n",
        "joblib.dump(scaler, 'scaler.pkl')\n",
        "print(\"Saved scaler.pkl\")\n",
        "\n",
        "# Save model\n",
        "joblib.dump(best_xgb, 'xgb_model.joblib')\n",
        "print(\"Saved xgb_model.joblib\")"
    ]
}

nb['cells'].append(new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("Notebook updated successfully.")
