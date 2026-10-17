# APS-IA — Entrega de pacote com A*

APS de Inteligência Artificial e Robótica: um agente deve buscar um pacote e
depositá-lo no destino em um grid quadrado, minimizando o custo total. Há
obstáculos e pedágios; mundos sem solução também são aceitos.

## Estrutura e execução

O repositório versiona exatamente três arquivos:

```text
main.py       # domínio, A*, geração, Tkinter e testes internos
README.md     # documentação
.gitignore    # exclusão de caches e arquivos locais
```

Recomendado: Python 3.11 ou superior com Tkinter. Validado com Python 3.14.4.
Usa apenas a biblioteca-padrão, sem instalação de dependências externas.
No Windows PowerShell, dentro da pasta do projeto:

```powershell
python --version
python main.py
python main.py --test
```

O primeiro comando mostra a versão, o segundo abre a interface e o terceiro
executa os testes sem importar nem abrir o Tkinter. A distribuição oficial do
Python para Windows oferece Tcl/Tk no instalador. Em sistemas sem suporte
gráfico, a busca e os testes continuam disponíveis; a interface exige Tkinter
e uma sessão gráfica.

## Modelagem e correlação com a prova

A modelagem deriva diretamente da resposta apresentada na avaliação:

```python
estado = (posicao_agente, estado_pacote)
posicao_agente = (linha, coluna)
```

O pacote é uma posição `(linha, coluna)` quando está depositado, ou a string
`"carregando"` quando está com o agente. As coordenadas começam em zero.

Exemplos:

- `((0, 0), (1, 2))`: agente em `(0, 0)` e pacote em `(1, 2)`.
- `((1, 2), "carregando")`: agente carregando o pacote.
- `((3, 3), (3, 3))`: pacote depositado junto ao agente.

O estado inicial contém as posições sorteadas do agente e do pacote, sempre
distintas entre si e do destino. Para cenários definidos diretamente no código,
o domínio também aceita agente e pacote juntos, pacote carregado ou meta inicial.

A condição de meta é exclusivamente `estado[1] == problema.destino`.
Estar no destino carregando o pacote ainda exige ENTREGAR. Não há marcador
`"entregue"`: uma entrega fora do destino mantém a posição real do pacote e
permite pegá-lo novamente.

Dimensão, obstáculos, pedágios e destino são dados fixos da classe `Problema`.
Eles não fazem parte da tupla dinâmica. As tuplas dos estados são imutáveis e
podem ser usadas como chaves de dicionário; os conjuntos do mundo são `frozenset`.

## Ações, sucessores e custos

| Ação | Condição e efeito |
| --- | --- |
| CIMA | Move `(-1, 0)` se a célula for livre e estiver dentro do grid. |
| BAIXO | Move `(+1, 0)` nas mesmas condições. |
| ESQUERDA | Move `(0, -1)` nas mesmas condições. |
| DIREITA | Move `(0, +1)` nas mesmas condições. |
| PEGAR | Exige pacote depositado na célula do agente; troca o pacote por `"carregando"`. |
| ENTREGAR | Exige `"carregando"`; deposita o pacote na célula atual, mesmo fora do destino. |

`Problema.sucessores(estado)` produz triplas `(novo_estado, acao, custo)` sem
alterar o estado pai. Ao mover carregando, a string permanece no estado e o
pacote acompanha a posição do agente.

Conceitualmente, tentar sair do grid ou atravessar obstáculo mantém o agente
na mesma célula. Esses auto-loops são omitidos na busca: retornam ao mesmo
estado com custo positivo e nunca melhoram uma solução ótima. Ações de pegar
ou entregar sem suas precondições também não são incluídas.

Toda ação custa **1**. Se a célula de **origem**, antes da ação, for pedágio,
custa **3** (`1 + 2`). Isso vale também para PEGAR e ENTREGAR. Entrar em um
pedágio vindo de uma célula livre custa 1; sair dele custa 3.

## Busca A* e heurística

`a_estrela` implementa busca em grafo com `f(n) = g(n) + h(n)`, usando `heapq`.
`melhor_g[estado]` armazena o menor custo conhecido; só uma melhora estrita
reinsere um estado. Entradas antigas do heap são descartadas. Um contador
desempata prioridades sem comparar estados que misturam tuplas e strings.
Pais e ações permitem reconstruir o plano quando a meta é retirada do heap.

O resultado contém existência de solução, lista de ações, lista de estados
(incluindo o inicial), custo total e estados expandidos. A contagem considera
as retiradas válidas cujos sucessores são gerados; a meta não é expandida.
Se a fronteira esvaziar, o resultado tem `existe=False`, listas vazias e
`custo=None`. A interface informa: **Não existe solução para este cenário.**

Com `M` representando a distância Manhattan:

```text
Pacote no destino:       h = 0
Pacote carregado:        h = M(agente, destino) + 1
Pacote depositado:       h = M(agente, pacote) + 1 + M(pacote, destino) + 1
```

Os termos adicionais representam PEGAR e ENTREGAR. A heurística é admissível:
ignora obstáculos, desvios e adicionais dos pedágios. As distâncias são limites
inferiores dos movimentos necessários e cada ação custa pelo menos 1. Portanto,
não superestima o custo ótimo restante. Entregar provisoriamente fora do destino
não elimina a necessidade de pegar e transportar o pacote até a meta.

O espaço de estados é finito e os custos são positivos. Com essa heurística e
a atualização de menores custos, A* retorna um plano de custo mínimo. A função
aceita uma heurística alternativa; nos testes, `lambda estado: 0` transforma
a busca em Dijkstra para comparar os custos ótimos.

## Geração aleatória

`gerar_mundo(N, seed)` usa seu próprio `random.Random(seed)` e exige `N >= 2`.
Uma lista embaralhada de células fornece três posições distintas para agente,
pacote e destino. Os obstáculos são sorteados entre as demais células.

- Obstáculos: quantidade inteira aleatória entre `floor(10% * disponíveis)`
  e `floor(25% * disponíveis)`, inclusive, após reservar as três posições.
- Pedágios: quantidade inteira aleatória entre `floor(5% * restantes)` e
  `floor(15% * restantes)`, inclusive, considerando todas as células sem obstáculo.

Arredondar para baixo permite zero obstáculos ou pedágios em grids pequenos.
O sorteio é sem reposição. Pedágios podem coincidir com agente, pacote ou destino,
mas nunca com obstáculos. Nenhuma conectividade é imposta: cenários sem solução
são possíveis. A mesma dimensão e seed reproduzem o mundo na mesma versão do
Python. Detalhes de embaralhamento podem variar entre versões do interpretador.

## Interface

Informe N e uma seed inteira opcional e clique em **Gerar mundo**. Uma seed
vazia é sorteada e exibida para permitir reprodução. **Outro cenário** escolhe
uma nova seed. A interface aceita dimensões entre 2 e 50 para limitar o desenho;
a função de geração não impõe esse máximo.

Clique em **Resolver A*** para obter o plano, exibido na lista lateral com o
custo de cada ação. A busca roda em uma thread para manter a janela responsiva;
a comunicação com Tkinter usa uma fila consultada por `after()`. Gerar um novo
mundo cancela a busca anterior e descarta resultados antigos.

Use **Animar / Pausar**, **Avançar um passo** ou **Reiniciar** para inspecionar
o plano. Reiniciar retorna ao estado inicial do mesmo mundo e preserva o plano.
A animação usa `after(450, ...)`, sem `time.sleep()`. O painel mostra ação atual,
custo acumulado, custo total, progresso das ações e estados expandidos.

O Canvas desenha células livres brancas, obstáculos pretos, pedágios cinza com
`$`, agente azul `A`, pacote laranja `P` e destino com borda verde `D`.
`A+P` representa o agente carregando. A legenda fica abaixo do desenho.

## Exemplos reais e validação

Resultados obtidos executando o código com Python 3.14.4:

1. Mundo fixo 3×3 sem obstáculos ou pedágios, agente `(0, 0)`, pacote `(1, 1)`
   e destino `(2, 2)`: plano `BAIXO, DIREITA, PEGAR, BAIXO, DIREITA, ENTREGAR`;
   custo **6**, **6 ações**, **8 estados expandidos**.
2. Na interface, N **5**, seed **42**: agente `(3, 1)`, pacote `(2, 2)` e
   destino `(1, 4)`. Obstáculos `{(1, 0), (1, 1), (3, 3), (3, 4)}` e pedágio
   `{(4, 1)}`. Plano `CIMA, DIREITA, PEGAR, CIMA, DIREITA, DIREITA, ENTREGAR`;
   custo **7**, **7 ações**, **10 estados expandidos**.
3. Mundo fixo 3×3 com agente `(0, 0)`, pacote `(1, 1)`, destino `(2, 2)` e
   obstáculos em `(0, 1)` e `(1, 0)`: sem solução, **1 estado expandido**.
4. O teste de rota com pedágio usa grid 5×5, agente e pacote em `(1, 0)` e
   destino `(1, 4)`. Sem pedágios, o custo é **6**. Com pedágios em `(1, 1)`,
   `(1, 2)` e `(1, 3)`, o plano ótimo desvia e custa **8**.

Os cenários fixos estão nos testes internos; o segundo exemplo pode ser
reproduzido diretamente pelos campos da interface. Para consultar o primeiro
no PowerShell sem abrir a janela:

```powershell
python -c "from main import Problema, a_estrela; p = Problema(3, frozenset(), frozenset(), (2, 2), ((0, 0), (1, 1))); print(a_estrela(p))"
```

`python main.py --test` executou **24 testes**, todos aprovados. Cobrem movimentos,
borda, obstáculos, pegar, transporte, entregas, meta, todos os custos, heurística,
solução e ausência de solução, rota alterada por pedágio, geração sem sobreposições,
seed, dimensões, meta inicial e cancelamento. Incluem comparação A*/Dijkstra
em **75 mundos** (dimensões 2, 3 e 4; seeds de 0 a 24).

A sintaxe foi conferida por `ast.parse`. A interface foi aberta e exercitada:
geração, reprodução por seed, resolução em thread, avanço, animação, reinício,
conclusão do plano, dimensões 2, 5, 8 e 15 e mensagem de ausência de solução.

## Decisões e limitações

Todo o domínio, busca, interface e testes permanecem em `main.py`, organizados
em classes e funções, para manter a entrega simples. Tkinter é importado somente
ao iniciar a interface. Não são usadas imagens nem arquivos auxiliares.

A* mantém a fronteira, custos e predecessores em memória. O espaço de estados
pode crescer na ordem de N⁴, pois o pacote pode ser depositado em diferentes
células. Mundos grandes ou sem solução podem exigir muita memória e tempo;
a thread mantém a interface responsiva, mas não reduz esse consumo. Não há
editor manual de cenários, persistência em disco ou limite automático de busca.
Grids muito densos ficam pequenos visualmente. O objetivo é minimizar custo,
que pode favorecer mais movimentos para evitar pedágios.

## Git e GitHub

Somente `main.py`, `README.md` e `.gitignore` devem ser versionados. Caches,
ambientes virtuais e arquivos comuns do editor são ignorados.

O projeto já possui o remoto `origin` configurado. Confira-o antes de publicar:

```powershell
git remote -v
git status
gh auth login
git push -u origin HEAD
```

O push deve ser executado após a confirmação do responsável pelo projeto.
Se usar outro repositório, crie um repositório vazio na sua conta pelo GitHub,
escolha a visibilidade desejada e substitua o remoto antes do push:

```powershell
git remote set-url origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin HEAD
```

A publicação não faz parte da execução do programa. O envio do link ao
Blackboard fica a cargo do aluno.
