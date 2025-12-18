Antes de executar o projeto, devem ser instaladas as dependências através do ficheiro de requisitos usando `pip install -r requirements.txt`.

Para efetuar a comparação dos modelos, basta executar o ficheiro de comparação, não sendo necessário passar qualquer argumento. O script carrega automaticamente os quatro modelos definidos e apresenta os resultados finais. Para isso, basta correr `python comparison_models.py`.

Para efetuar a visualização das simulações, utiliza-se o ficheiro `evaluate_model.py`, sendo necessário executar um comando por cada modelo existente. Assim, devem ser corridos os seguintes comandos:

`python evaluate_model.py --env original --model final_models/ppo_original.zip --episodes 1 --algo ppo`

`python evaluate_model.py --env custom --model final_models/ppo_custom.zip --episodes 1 --algo ppo`

`python evaluate_model.py --env zoo --model final_models/ppo_zoo_custom.zip --episodes 1 --algo ppo`

`python evaluate_model.py --env custom --model final_models/sac_custom.zip --episodes 1 --algo sac`

O número de episódios (--episodes), que neste exemplo está definido como 1, pode ser ajustado conforme pretendido.

Cada comando permite visualizar o comportamento do respetivo agente no ambiente correspondente.
