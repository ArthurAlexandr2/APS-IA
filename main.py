"""APS: entrega de pacote com A*, interface Tkinter e testes internos."""

import argparse
from dataclasses import dataclass
import heapq
from itertools import count
import math
import queue
import random
import threading
from time import perf_counter
from typing import Callable, Iterator
import unittest

Posicao = tuple[int, int]
Estado = tuple[Posicao, Posicao | str]
CARREGANDO = "carregando"
MOVIMENTOS = {"CIMA": (-1, 0), "BAIXO": (1, 0),
              "ESQUERDA": (0, -1), "DIREITA": (0, 1)}


def manhattan(a: Posicao, b: Posicao) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class Problema:
    """Dados fixos; a tupla dinâmica deriva da resposta da avaliação."""

    dimensao: int
    obstaculos: frozenset[Posicao]
    pedagios: frozenset[Posicao]
    destino: Posicao
    inicial: Estado

    def __post_init__(self):
        if type(self.dimensao) is not int or self.dimensao < 2:
            raise ValueError("A dimensão deve ser pelo menos 2.")
        # Copia também conjuntos recebidos pelo chamador: frozen=True sozinho
        # impede reatribuição, mas não impede alterações dentro de um set.
        object.__setattr__(self, "obstaculos", frozenset(self.obstaculos))
        object.__setattr__(self, "pedagios", frozenset(self.pedagios))
        posicoes = set(self.obstaculos) | set(self.pedagios) | {self.destino, self.inicial[0]}
        pacote = self.inicial[1]
        if pacote != CARREGANDO:
            posicoes.add(pacote)
        if any(not self.dentro(p) for p in posicoes):
            raise ValueError("Posição fora do grid.")
        if self.obstaculos & (self.pedagios | {self.destino, self.inicial[0]}):
            raise ValueError("Obstáculo em uma posição reservada.")
        if pacote != CARREGANDO and pacote in self.obstaculos:
            raise ValueError("Pacote sobre obstáculo.")

    def dentro(self, p: Posicao) -> bool:
        return 0 <= p[0] < self.dimensao and 0 <= p[1] < self.dimensao

    def is_goal(self, estado: Estado) -> bool:
        return estado[1] == self.destino

    def custo(self, estado: Estado) -> int:
        return 3 if estado[0] in self.pedagios else 1

    def sucessores(self, estado: Estado) -> Iterator[tuple[Estado, str, int]]:
        agente, pacote = estado
        custo = self.custo(estado)
        for acao, (dl, dc) in MOVIMENTOS.items():
            novo = (agente[0] + dl, agente[1] + dc)
            if self.dentro(novo) and novo not in self.obstaculos:
                yield (novo, pacote), acao, custo
        if pacote == agente:
            yield (agente, CARREGANDO), "PEGAR", custo
        if pacote == CARREGANDO:
            yield (agente, agente), "ENTREGAR", custo

    def heuristica(self, estado: Estado) -> int:
        agente, pacote = estado
        if self.is_goal(estado):
            return 0
        if pacote == CARREGANDO:
            return manhattan(agente, self.destino) + 1
        return manhattan(agente, pacote) + 1 + manhattan(pacote, self.destino) + 1


@dataclass(frozen=True)
class EstadoEntrega:
    """Nó no estilo dos exemplos da disciplina, sem dependências externas.

    estado é a tupla da prova; op e valor_op descrevem a transição de entrada.
    O custo acumulado pertence à busca, não à identidade do estado.
    """

    problema: Problema
    estado: Estado
    op: str = ""
    valor_op: int = 0

    def successors(self) -> list["EstadoEntrega"]:
        return [EstadoEntrega(self.problema, novo, acao, custo)
                for novo, acao, custo in self.problema.sucessores(self.estado)]

    def is_goal(self) -> bool:
        return self.problema.is_goal(self.estado)

    def description(self) -> str:
        return "Entrega de pacote em grid com obstáculos e pedágios"

    def cost(self) -> int:
        """Custo da ação que gerou este nó, calculado na célula do pai."""
        return self.valor_op

    def env(self) -> Estado:
        """Identidade dinâmica; não inclui op, custo ou dados fixos do mundo."""
        return self.estado

    def h(self) -> int:
        return self.problema.heuristica(self.estado)

    def __str__(self) -> str:
        return f"Operação: {self.op or 'INÍCIO'} | Estado: {self.estado}"


@dataclass
class Resultado:
    existe: bool
    acoes: list[str]
    estados: list[Estado]
    custo: int | None
    expandidos: int
    tempo_segundos: float = 0.0

    def __bool__(self) -> bool:
        return self.existe

    @property
    def g(self) -> int | None:
        """Nome usado nos exemplos para o custo acumulado da solução."""
        return self.custo

    def show_path(self) -> str:
        if not self.existe:
            return "Não existe solução para este cenário."
        linhas = [f"INÍCIO: {self.estados[0]}"]
        linhas.extend(f"{i}. {acao}: {estado}"
                      for i, (acao, estado) in enumerate(zip(self.acoes, self.estados[1:]), 1))
        linhas.append(f"Custo total: {self.g}")
        return "\n".join(linhas)


class AEstrela:
    """Busca em grafo com a interface algorithm.search(state) dos exemplos."""

    def search(self, inicial: EstadoEntrega,
               heuristica: Callable[[Estado], int] | None = None,
               cancelar: threading.Event | None = None) -> Resultado:
        inicio = perf_counter()

        def h(no):
            return no.h() if heuristica is None else heuristica(no.env())

        desempate = count()
        fronteira = [(h(inicial), next(desempate), 0, inicial)]
        melhor_g = {inicial.env(): 0}
        pais: dict[Estado, tuple[Estado, str]] = {}
        expandidos = 0
        while fronteira:
            if cancelar is not None and cancelar.is_set():
                raise InterruptedError("Busca cancelada.")
            _, _, g, no = heapq.heappop(fronteira)
            estado = no.env()
            if g != melhor_g[estado]:
                continue
            if no.is_goal():
                estados, acoes = [estado], []
                while estado in pais:
                    estado, acao = pais[estado]
                    estados.append(estado)
                    acoes.append(acao)
                return Resultado(True, acoes[::-1], estados[::-1], g, expandidos,
                                 perf_counter() - inicio)
            expandidos += 1
            for sucessor in no.successors():
                novo = sucessor.env()
                novo_g = g + sucessor.cost()
                if novo_g < melhor_g.get(novo, math.inf):
                    melhor_g[novo] = novo_g
                    pais[novo] = (estado, sucessor.op)
                    heapq.heappush(fronteira, (novo_g + h(sucessor), next(desempate), novo_g, sucessor))
        return Resultado(False, [], [], None, expandidos, perf_counter() - inicio)


def a_estrela(problema: Problema,
              heuristica: Callable[[Estado], int] | None = None,
              cancelar: threading.Event | None = None) -> Resultado:
    """Atalho usado pela interface e pelos testes; h=0 executa Dijkstra."""
    return AEstrela().search(EstadoEntrega(problema, problema.inicial), heuristica, cancelar)


def gerar_mundo(dimensao: int, seed: int | None = None) -> Problema:
    if type(dimensao) is not int or dimensao < 2:
        raise ValueError("A dimensão deve ser pelo menos 2.")
    rng = random.Random(seed)
    celulas = [(l, c) for l in range(dimensao) for c in range(dimensao)]
    rng.shuffle(celulas)
    agente, pacote, destino = celulas[:3]
    disponiveis = celulas[3:]
    # Arredondamento para baixo permite quantidades zero em grids pequenos.
    qtd = rng.randint(int(.10 * len(disponiveis)), int(.25 * len(disponiveis)))
    obstaculos = frozenset(disponiveis[:qtd])
    restantes = [p for p in celulas if p not in obstaculos]
    qtd_p = rng.randint(int(.05 * len(restantes)), int(.15 * len(restantes)))
    pedagios = frozenset(rng.sample(restantes, qtd_p))
    return Problema(dimensao, obstaculos, pedagios, destino, (agente, pacote))


class Interface:
    def __init__(self, raiz):
        import tkinter as tk
        from tkinter import ttk
        self.raiz = raiz
        raiz.title("APS — Entrega de pacote com A*")
        raiz.geometry("1020x780")
        self.problema = None
        self.resultado = None
        self.indice = 0
        self.custo_atual = 0
        self.animacao = None
        self.buscando = False
        self.consulta = None
        self.cancelar = threading.Event()
        self.geracao = 0
        self.fila = queue.Queue()
        topo = ttk.Frame(raiz, padding=8)
        topo.pack(fill="x")
        ttk.Label(topo, text="Dimensão N:").pack(side="left")
        self.n = tk.StringVar(value="8")
        ttk.Entry(topo, textvariable=self.n, width=6).pack(side="left", padx=5)
        ttk.Label(topo, text="Seed (opcional):").pack(side="left")
        self.seed = tk.StringVar()
        ttk.Entry(topo, textvariable=self.seed, width=12).pack(side="left", padx=5)
        ttk.Button(topo, text="Gerar mundo", command=self.gerar).pack(side="left", padx=4)
        ttk.Button(topo, text="Outro cenário", command=lambda: self.gerar(True)).pack(side="left")
        barra = ttk.Frame(raiz, padding=8)
        barra.pack(fill="x")
        self.resolver_botao = ttk.Button(barra, text="Resolver A*", command=self.resolver)
        self.resolver_botao.pack(side="left", padx=4)
        self.animar_botao = ttk.Button(barra, text="Animar / Pausar", command=self.animar)
        self.avancar_botao = ttk.Button(barra, text="Avançar um passo", command=self.avancar)
        self.reiniciar_botao = ttk.Button(barra, text="Reiniciar", command=self.reiniciar)
        for botao in (self.animar_botao, self.avancar_botao, self.reiniciar_botao):
            botao.pack(side="left", padx=4)
        self.status = tk.StringVar()
        ttk.Label(raiz, textvariable=self.status, wraplength=980, padding=8).pack(fill="x")
        self.progresso = ttk.Progressbar(raiz, mode="determinate")
        self.progresso.pack(fill="x", padx=10)
        corpo = ttk.Frame(raiz)
        corpo.pack(fill="both", expand=True, padx=10, pady=8)
        self.canvas = tk.Canvas(corpo, bg="white", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        painel = ttk.Frame(corpo)
        painel.pack(side="right", fill="y", padx=(10, 0))
        ttk.Label(painel, text="Plano: ação e custo").pack()
        self.plano = tk.Listbox(painel, width=28, exportselection=False)
        scroll = ttk.Scrollbar(painel, orient="vertical", command=self.plano.yview)
        self.plano.configure(yscrollcommand=scroll.set)
        self.plano.pack(side="left", fill="y")
        scroll.pack(side="right", fill="y")
        ttk.Label(raiz, padding=8, text="Legenda: branco = livre | preto = obstáculo | cinza = pedágio ($) | "
                  "azul A = agente | laranja P = pacote | verde D = destino | A+P = carregando").pack()
        self.canvas.bind("<Configure>", lambda evento: self.desenhar())
        raiz.protocol("WM_DELETE_WINDOW", self.fechar)
        self.gerar()
        self.consulta = raiz.after(100, self.receber)

    def atualizar_controles(self):
        r = self.resultado
        pode_reproduzir = not self.buscando and r is not None and r.existe
        tem_passos = pode_reproduzir and self.indice < len(r.acoes)
        self.resolver_botao.state(["disabled" if self.buscando else "!disabled"])
        self.reiniciar_botao.state(["!disabled" if pode_reproduzir else "disabled"])
        for botao in (self.animar_botao, self.avancar_botao):
            botao.state(["!disabled" if tem_passos else "disabled"])

    def parar(self):
        if self.animacao is not None:
            self.raiz.after_cancel(self.animacao)
            self.animacao = None

    def gerar(self, outro=False):
        from tkinter import messagebox
        try:
            n = int(self.n.get())
            if not 2 <= n <= 50:
                raise ValueError("Na interface, use uma dimensão entre 2 e 50.")
            seed = random.SystemRandom().randrange(2**32) if outro or not self.seed.get().strip() else int(self.seed.get())
            problema = gerar_mundo(n, seed)
        except ValueError as erro:
            messagebox.showerror("Entrada inválida", str(erro))
            return
        self.parar()
        self.cancelar.set()
        self.geracao += 1
        self.seed.set(str(seed))
        self.problema = problema
        self.resultado = None
        self.buscando = False
        self.plano.delete(0, "end")
        self.reiniciar()

    def resolver(self):
        if self.buscando:
            return
        self.parar()
        self.cancelar.set()
        self.cancelar = threading.Event()
        self.geracao += 1
        token, problema, cancelar = self.geracao, self.problema, self.cancelar
        self.resultado = None
        self.indice, self.custo_atual = 0, 0
        self.plano.delete(0, "end")
        self.buscando = True
        self.atualizar_status()
        self.desenhar()

        def trabalho():
            try:
                resultado = a_estrela(problema, cancelar=cancelar)
                self.fila.put((token, resultado))
            except InterruptedError:
                pass
            except Exception as erro:
                self.fila.put((token, erro))
        threading.Thread(target=trabalho, daemon=True).start()

    def receber(self):
        try:
            while True:
                token, resultado = self.fila.get_nowait()
                if token != self.geracao:
                    continue
                self.buscando = False
                if isinstance(resultado, Exception):
                    self.atualizar_controles()
                    self.status.set(f"Erro na busca: {resultado}")
                    continue
                self.resultado = resultado
                self.plano.delete(0, "end")
                for i, acao in enumerate(resultado.acoes):
                    custo = self.problema.custo(resultado.estados[i])
                    self.plano.insert("end", f"{i + 1}. {acao}  (+{custo})")
                self.reiniciar()
        except queue.Empty:
            pass
        self.consulta = self.raiz.after(100, self.receber)

    def atualizar_status(self):
        r = self.resultado
        if self.buscando:
            self.status.set("Buscando plano ótimo… Gere outro mundo para cancelar a busca.")
        elif r is None:
            self.status.set("Mundo gerado. Clique em Resolver A*. Ação: início | custo acumulado: 0")
        elif not r.existe:
            self.status.set(f"Não existe solução para este cenário. Estados expandidos: {r.expandidos} | "
                            f"tempo de busca: {r.tempo_segundos:.4f} s")
        else:
            acao = r.acoes[self.indice - 1] if self.indice else "início"
            fim = " | Meta alcançada!" if self.problema.is_goal(r.estados[self.indice]) else ""
            self.status.set(f"Ação atual: {acao} | custo acumulado: {self.custo_atual} | custo total: {r.custo} | "
                            f"ações: {self.indice}/{len(r.acoes)} | estados expandidos: {r.expandidos} | "
                            f"tempo de busca: {r.tempo_segundos:.4f} s{fim}")
        self.progresso.configure(maximum=max(1, len(r.acoes) if r else 1), value=self.indice)
        self.atualizar_controles()

    def reiniciar(self):
        if self.buscando:
            return
        self.parar()
        self.indice, self.custo_atual = 0, 0
        self.plano.selection_clear(0, "end")
        self.atualizar_status()
        self.desenhar()

    def avancar(self):
        self.parar()
        self.passo()

    def passo(self):
        r = self.resultado
        if self.buscando or r is None or not r.existe or self.indice >= len(r.acoes):
            return False
        self.custo_atual += self.problema.custo(r.estados[self.indice])
        self.indice += 1
        self.plano.selection_clear(0, "end")
        self.plano.selection_set(self.indice - 1)
        self.plano.see(self.indice - 1)
        self.atualizar_status()
        self.desenhar()
        return self.indice < len(r.acoes)

    def animar(self):
        if self.animacao is not None:
            self.parar()
            return
        self.tick()

    def tick(self):
        self.animacao = None
        if self.passo():
            self.animacao = self.raiz.after(450, self.tick)

    def desenhar(self):
        if self.problema is None:
            return
        p, c = self.problema, self.canvas
        c.delete("all")
        lado = max(1, min(c.winfo_width(), c.winfo_height()) - 12) / p.dimensao
        fonte = max(5, min(16, int(lado * .28)))
        estado = self.resultado.estados[self.indice] if self.resultado and self.resultado.existe else p.inicial
        agente, pacote = estado
        for l in range(p.dimensao):
            for col in range(p.dimensao):
                pos = (l, col)
                x, y = 6 + col * lado, 6 + l * lado
                cor = "#151515" if pos in p.obstaculos else "#b4b4b4" if pos in p.pedagios else "white"
                c.create_rectangle(x, y, x + lado, y + lado, fill=cor, outline="#d0d0d0")
                if pos == p.destino:
                    c.create_rectangle(x + 2, y + 2, x + lado - 2, y + lado - 2, outline="#008844", width=3)
                if pos in p.pedagios:
                    c.create_text(x + lado * .15, y + lado * .18, text="$", font=("Arial", fonte))
                if pos == agente:
                    c.create_oval(x + lado * .18, y + lado * .18, x + lado * .82, y + lado * .82, fill="#276cce", outline="")
                    c.create_text(x + lado / 2, y + lado / 2, text="A+P" if pacote == CARREGANDO else "A", fill="white", font=("Arial", fonte, "bold"))
                if pos == pacote:
                    c.create_rectangle(x + lado * .62, y + lado * .60, x + lado * .94, y + lado * .94, fill="#ff9c27", outline="")
                    c.create_text(x + lado * .78, y + lado * .77, text="P", font=("Arial", fonte))
                if pos == p.destino:
                    c.create_text(x + lado * .83, y + lado * .15, text="D", fill="#008844", font=("Arial", fonte, "bold"))

    def fechar(self):
        self.cancelar.set()
        self.parar()
        if self.consulta is not None:
            self.raiz.after_cancel(self.consulta)
            self.consulta = None
        self.raiz.destroy()


class Testes(unittest.TestCase):
    def problema(self, obstaculos=(), pedagios=(), destino=(2, 2), inicial=((0, 0), (1, 1)), n=3):
        return Problema(n, frozenset(obstaculos), frozenset(pedagios), destino, inicial)

    def acoes(self, p, estado):
        return {acao: (novo, custo) for novo, acao, custo in p.sucessores(estado)}

    def test_movimento_valido(self):
        p = self.problema()
        self.assertEqual(self.acoes(p, p.inicial)["DIREITA"][0], ((0, 1), (1, 1)))

    def test_borda(self):
        p = self.problema()
        self.assertNotIn("CIMA", self.acoes(p, p.inicial))
        self.assertNotIn("ESQUERDA", self.acoes(p, p.inicial))

    def test_obstaculo(self):
        p = self.problema(obstaculos=[(0, 1)])
        self.assertNotIn("DIREITA", self.acoes(p, p.inicial))

    def test_pegar(self):
        p = self.problema()
        self.assertEqual(self.acoes(p, ((1, 1), (1, 1)))["PEGAR"][0], ((1, 1), CARREGANDO))

    def test_pegar_distancia(self):
        p = self.problema()
        self.assertNotIn("PEGAR", self.acoes(p, p.inicial))

    def test_movimento_carregando(self):
        p = self.problema()
        self.assertEqual(self.acoes(p, ((0, 0), CARREGANDO))["DIREITA"][0], ((0, 1), CARREGANDO))

    def test_entregar_fora(self):
        p = self.problema()
        novo, _ = self.acoes(p, ((1, 1), CARREGANDO))["ENTREGAR"]
        self.assertEqual(novo, ((1, 1), (1, 1)))
        self.assertFalse(p.is_goal(novo))
        self.assertIn("PEGAR", self.acoes(p, novo))

    def test_entregar_destino(self):
        p = self.problema()
        novo, _ = self.acoes(p, ((2, 2), CARREGANDO))["ENTREGAR"]
        self.assertTrue(p.is_goal(novo))

    def test_meta(self):
        p = self.problema()
        self.assertTrue(p.is_goal(((0, 0), (2, 2))))
        self.assertFalse(p.is_goal(((2, 2), CARREGANDO)))

    def test_custo_normal(self):
        self.assertEqual(self.problema().custo(((0, 0), (1, 1))), 1)

    def test_custo_pedagio_origem(self):
        p = self.problema(pedagios=[(0, 1)])
        self.assertEqual(self.acoes(p, p.inicial)["DIREITA"][1], 1)
        self.assertEqual(self.acoes(p, ((0, 1), (1, 1)))["ESQUERDA"][1], 3)

    def test_pegar_pedagio(self):
        p = self.problema(pedagios=[(1, 1)])
        self.assertEqual(self.acoes(p, ((1, 1), (1, 1)))["PEGAR"][1], 3)

    def test_entregar_pedagio(self):
        p = self.problema(pedagios=[(2, 2)])
        self.assertEqual(self.acoes(p, ((2, 2), CARREGANDO))["ENTREGAR"][1], 3)

    def test_heuristica(self):
        p = self.problema()
        self.assertEqual(p.heuristica(((0, 0), (2, 2))), 0)
        self.assertEqual(p.heuristica(((0, 0), CARREGANDO)), 5)
        self.assertEqual(p.heuristica(p.inicial), 6)

    def test_solucao(self):
        p = self.problema()
        r = a_estrela(p)
        self.assertTrue(r.existe)
        self.assertEqual(r.custo, 6)
        self.assertIn("PEGAR", r.acoes)
        self.assertEqual(r.acoes[-1], "ENTREGAR")
        self.assertEqual(len(r.estados), len(r.acoes) + 1)
        self.assertEqual(r.estados[0], p.inicial)
        for i, acao in enumerate(r.acoes):
            self.assertIn((r.estados[i + 1], acao, p.custo(r.estados[i])), list(p.sucessores(r.estados[i])))

    def test_sem_solucao(self):
        p = self.problema(obstaculos=[(0, 1), (1, 0)])
        r = a_estrela(p)
        self.assertFalse(r.existe)
        self.assertIsNone(r.custo)
        self.assertEqual(r.acoes, [])

    def test_pedagio_altera_rota(self):
        inicial = ((1, 0), (1, 0))
        sem = self.problema(n=5, destino=(1, 4), inicial=inicial)
        com = self.problema(n=5, destino=(1, 4), inicial=inicial, pedagios=[(1, 1), (1, 2), (1, 3)])
        r0, r1 = a_estrela(sem), a_estrela(com)
        self.assertEqual(r0.custo, 6)
        self.assertEqual(r1.custo, 8)
        self.assertFalse(any(e[0] in com.pedagios for e in r1.estados))
        self.assertEqual(r1.custo, a_estrela(com, lambda e: 0).custo)

    def test_geracao_sem_sobreposicoes(self):
        for n in (2, 3, 8, 15):
            for seed in range(20):
                p = gerar_mundo(n, seed)
                reservadas = {p.inicial[0], p.inicial[1], p.destino}
                self.assertEqual(len(reservadas), 3)
                self.assertFalse(reservadas & p.obstaculos)
                self.assertFalse(p.pedagios & p.obstaculos)
                self.assertTrue(all(p.dentro(pos) for pos in reservadas | p.obstaculos | p.pedagios))

    def test_seed(self):
        self.assertEqual(gerar_mundo(8, 42), gerar_mundo(8, 42))

    def test_dimensoes(self):
        for n in (2, 3, 5, 8):
            p = gerar_mundo(n, 42)
            self.assertEqual(p.dimensao, n)
            self.assertEqual(a_estrela(p).custo, a_estrela(p, lambda e: 0).custo)
        with self.assertRaises(ValueError):
            gerar_mundo(1)

    def test_otimalidade_e_admissibilidade(self):
        for n in (2, 3, 4):
            for seed in range(25):
                p = gerar_mundo(n, seed)
                r, d = a_estrela(p), a_estrela(p, lambda e: 0)
                self.assertEqual((r.existe, r.custo), (d.existe, d.custo))
                if r.existe:
                    self.assertLessEqual(p.heuristica(p.inicial), r.custo)

    def test_meta_inicial(self):
        p = self.problema(inicial=((0, 0), (2, 2)))
        r = a_estrela(p)
        self.assertEqual((r.custo, r.acoes, r.estados, r.expandidos), (0, [], [p.inicial], 0))

    def test_entregar_sem_carregar(self):
        p = self.problema()
        self.assertNotIn("ENTREGAR", self.acoes(p, p.inicial))

    def test_cancelamento(self):
        evento = threading.Event()
        evento.set()
        with self.assertRaises(InterruptedError):
            a_estrela(self.problema(), cancelar=evento)

    def test_dados_fixos_imutaveis(self):
        obstaculos, pedagios = {(0, 1)}, {(1, 0)}
        p = Problema(3, obstaculos, pedagios, (2, 2), ((0, 0), (1, 1)))
        obstaculos.add((0, 0))
        pedagios.clear()
        self.assertEqual(p.obstaculos, frozenset({(0, 1)}))
        self.assertEqual(p.pedagios, frozenset({(1, 0)}))
        self.assertIsInstance(p.obstaculos, frozenset)
        self.assertIsInstance(p.pedagios, frozenset)

    def test_dimensao_invalida(self):
        for n in (0, 1, -2, 2.5, True, "3"):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    gerar_mundo(n)
                with self.assertRaises(ValueError):
                    self.problema(n=n)

    def test_sucessores_preservam_pai(self):
        p = self.problema()
        for estado in (p.inicial, ((1, 1), (1, 1)), ((1, 1), CARREGANDO)):
            antes = repr(estado)
            for novo, _, _ in p.sucessores(estado):
                hash(novo)
            self.assertEqual(repr(estado), antes)

    def test_otimalidade_por_relaxacao_independente(self):
        # Bellman: calcula distâncias até a meta por relaxação das arestas,
        # sem reutilizar o heap, melhor_g ou reconstrução do A*.
        for obstaculos in (frozenset(), frozenset({(0, 1), (1, 0)})):
            livres = [(l, c) for l in range(2) for c in range(2)
                      if (l, c) not in obstaculos]
            for pedagios in (frozenset(), frozenset({(0, 0), (1, 1)})):
                p = Problema(2, obstaculos, pedagios, (1, 1), ((0, 0), (1, 1)))
                estados = [(a, pacote) for a in livres for pacote in [*livres, CARREGANDO]]
                distancias = {e: 0 if p.is_goal(e) else math.inf for e in estados}
                for _ in range(len(estados)):
                    mudou = False
                    for e in estados:
                        for novo, _, custo in p.sucessores(e):
                            if custo + distancias[novo] < distancias[e]:
                                distancias[e] = custo + distancias[novo]
                                mudou = True
                    if not mudou:
                        break
                for e in estados:
                    with self.subTest(obstaculos=obstaculos, pedagios=pedagios, estado=e):
                        teste = Problema(2, obstaculos, pedagios, p.destino, e)
                        r = a_estrela(teste)
                        esperado = distancias[e]
                        self.assertEqual(r.existe, esperado != math.inf)
                        self.assertEqual(r.custo, None if esperado == math.inf else esperado)
                        self.assertLessEqual(p.heuristica(e), esperado)
                        for novo, _, custo in p.sucessores(e):
                            self.assertLessEqual(p.heuristica(e), custo + p.heuristica(novo))


class TestesInterface(unittest.TestCase):
    """Regressões do controle da interface, sem abrir uma janela nos testes."""

    def setUp(self):
        from unittest.mock import Mock
        self.ui = Interface.__new__(Interface)
        ui = self.ui
        ui.raiz = Mock()
        ui.problema = gerar_mundo(5, 42)
        ui.resultado = a_estrela(ui.problema)
        ui.indice, ui.custo_atual = 0, 0
        ui.buscando = False
        ui.animacao, ui.consulta = None, "consulta"
        ui.cancelar = threading.Event()
        ui.geracao = 0
        ui.fila = queue.Queue()
        ui.status, ui.progresso, ui.plano = Mock(), Mock(), Mock()
        ui.resolver_botao, ui.animar_botao = Mock(), Mock()
        ui.avancar_botao, ui.reiniciar_botao = Mock(), Mock()
        ui.desenhar = Mock()

    def test_controles_sem_plano(self):
        ui = self.ui
        ui.resultado = None
        ui.atualizar_status()
        for botao in (ui.animar_botao, ui.avancar_botao, ui.reiniciar_botao):
            botao.state.assert_called_with(["disabled"])
        ui.resolver_botao.state.assert_called_with(["!disabled"])

    def test_busca_bloqueia_reproducao_e_preserva_mensagem(self):
        from unittest.mock import patch
        ui = self.ui
        with patch("threading.Thread") as thread:
            ui.resolver()
            ui.resolver()  # Não pode criar uma segunda busca em paralelo.
        thread.assert_called_once()
        self.assertTrue(ui.buscando)
        self.assertIsNone(ui.resultado)
        for botao in (ui.animar_botao, ui.avancar_botao, ui.reiniciar_botao, ui.resolver_botao):
            botao.state.assert_called_with(["disabled"])
        ui.status.set.reset_mock()
        ui.reiniciar()
        ui.avancar()
        ui.animar()
        ui.status.set.assert_not_called()
        self.assertEqual(ui.indice, 0)
        self.assertIsNone(ui.animacao)

    def test_receber_descarta_resultado_antigo(self):
        ui = self.ui
        r = ui.resultado
        ui.resultado, ui.buscando, ui.geracao = None, True, 2
        ui.fila.put((1, r))
        ui.receber()
        self.assertIsNone(ui.resultado)
        self.assertTrue(ui.buscando)
        ui.fila.put((2, r))
        ui.receber()
        self.assertIs(ui.resultado, r)
        self.assertFalse(ui.buscando)
        ui.avancar_botao.state.assert_called_with(["!disabled"])

    def test_erro_da_busca_libera_resolver(self):
        ui = self.ui
        ui.resultado, ui.buscando = None, True
        ui.fila.put((ui.geracao, RuntimeError("falha de teste")))
        ui.receber()
        self.assertFalse(ui.buscando)
        ui.resolver_botao.state.assert_called_with(["!disabled"])
        ui.animar_botao.state.assert_called_with(["disabled"])
        ui.status.set.assert_called_with("Erro na busca: falha de teste")

    def test_fim_do_plano_e_reinicio(self):
        ui = self.ui
        while ui.passo():
            pass
        self.assertEqual(ui.custo_atual, ui.resultado.custo)
        ui.animar_botao.state.assert_called_with(["disabled"])
        ui.avancar_botao.state.assert_called_with(["disabled"])
        ui.reiniciar_botao.state.assert_called_with(["!disabled"])
        ui.reiniciar()
        self.assertEqual((ui.indice, ui.custo_atual), (0, 0))
        ui.avancar_botao.state.assert_called_with(["!disabled"])

    def test_fechar_cancela_callbacks_e_busca(self):
        from unittest.mock import call
        ui = self.ui
        ui.animacao = "animacao"
        ui.fechar()
        self.assertTrue(ui.cancelar.is_set())
        self.assertEqual(ui.raiz.after_cancel.call_args_list, [call("animacao"), call("consulta")])
        ui.raiz.destroy.assert_called_once()


class TestesReferencia(unittest.TestCase):
    def setUp(self):
        self.problema = Problema(3, frozenset(), frozenset({(0, 0), (1, 1)}),
                                 (2, 2), ((0, 0), (1, 1)))
        self.state = EstadoEntrega(self.problema, self.problema.inicial)

    def test_metodos_do_estado(self):
        state = self.state
        self.assertEqual(state.env(), self.problema.inicial)
        self.assertEqual(state.h(), 6)
        self.assertFalse(state.is_goal())
        self.assertEqual(state.cost(), 0)  # Nó inicial não tem ação de entrada.
        self.assertIn("pacote", state.description())
        self.assertIn("INÍCIO", str(state))
        self.assertEqual({s.op for s in state.successors()}, {"BAIXO", "DIREITA"})
        self.assertTrue(all(s.cost() == 3 for s in state.successors()))
        self.assertTrue(all(s.problema is self.problema for s in state.successors()))
        self.assertEqual(state.env(), self.problema.inicial)

    def test_env_distingue_pacote_e_ignora_operacao(self):
        outro_caminho = EstadoEntrega(self.problema, self.state.estado, "CIMA", 3)
        self.assertEqual(self.state.env(), outro_caminho.env())
        carregando = EstadoEntrega(self.problema, (self.state.estado[0], CARREGANDO))
        self.assertNotEqual(self.state.env(), carregando.env())
        self.assertEqual(hash(self.state.env()), hash(outro_caminho.env()))

    def test_custo_da_transicao_de_origem(self):
        p = self.problema
        livre = EstadoEntrega(p, ((0, 1), (1, 1)))
        entrada = next(s for s in livre.successors() if s.op == "BAIXO")
        self.assertEqual(entrada.cost(), 1)  # Entrou em pedágio vindo de célula livre.
        pegou = next(s for s in entrada.successors() if s.op == "PEGAR")
        self.assertEqual((pegou.estado, pegou.cost()), (((1, 1), CARREGANDO), 3))
        entregou = next(s for s in pegou.successors() if s.op == "ENTREGAR")
        self.assertEqual(entregou.cost(), 3)
        self.assertFalse(entregou.is_goal())

    def test_algorithm_search_show_path_e_g(self):
        p = Problema(3, frozenset(), frozenset(), (2, 2), ((0, 0), (1, 1)))
        result = AEstrela().search(EstadoEntrega(p, p.inicial))
        self.assertTrue(result)
        self.assertEqual(result.g, 6)
        self.assertGreaterEqual(result.tempo_segundos, 0)
        plano = result.show_path()
        self.assertIn("INÍCIO: ((0, 0), (1, 1))", plano)
        self.assertIn("3. PEGAR: ((1, 1), 'carregando')", plano)
        self.assertIn("6. ENTREGAR: ((2, 2), (2, 2))", plano)
        self.assertIn("Custo total: 6", plano)

    def test_search_usa_no_inicial_recebido(self):
        state = EstadoEntrega(self.problema, ((2, 2), CARREGANDO))
        result = AEstrela().search(state)
        self.assertEqual(result.acoes, ["ENTREGAR"])
        self.assertEqual(result.estados[0], state.estado)
        self.assertEqual(result.g, 1)

    def test_resultado_sem_solucao(self):
        p = Problema(3, frozenset({(0, 1), (1, 0)}), frozenset(),
                     (2, 2), ((0, 0), (1, 1)))
        result = AEstrela().search(EstadoEntrega(p, p.inicial))
        self.assertFalse(result)
        self.assertIsNone(result.g)
        self.assertGreaterEqual(result.tempo_segundos, 0)
        self.assertEqual(result.show_path(), "Não existe solução para este cenário.")


def executar_cenarios():
    """Execução textual semelhante aos exemplos de cadeado e blocos."""
    cenarios = [
        ("Cenário 1 — sem obstáculos ou pedágios",
         Problema(3, frozenset(), frozenset(), (2, 2), ((0, 0), (1, 1)))),
        ("Cenário 2 — N=5, seed=42", gerar_mundo(5, 42)),
        ("Cenário 3 — sem solução",
         Problema(3, frozenset({(0, 1), (1, 0)}), frozenset(), (2, 2), ((0, 0), (1, 1)))),
    ]
    print("Busca A Estrela — Execução Detalhada")
    for nome, problema in cenarios:
        print(f"\n{'=' * 15} {nome} {'=' * 15}")
        state = EstadoEntrega(problema, problema.inicial)
        algorithm = AEstrela()
        result = algorithm.search(state)
        print(state.description())
        print(f"Início: {state.env()} | Destino: {problema.destino}")
        print(result.show_path())
        print(f"Estados expandidos: {result.expandidos}")
        print(f"Tempo de execução da busca: {result.tempo_segundos:.4f} segundos")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modos = parser.add_mutually_exclusive_group()
    modos.add_argument("--test", action="store_true", help="executar testes sem abrir a interface")
    modos.add_argument("--demo", action="store_true", help="mostrar cenários e planos no terminal")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=["main.py"], verbosity=2)
    elif args.demo:
        executar_cenarios()
    else:
        import tkinter as tk
        try:
            raiz = tk.Tk()
        except tk.TclError as erro:
            parser.exit(1, f"Não foi possível abrir o Tkinter: {erro}\n")
        Interface(raiz)
        raiz.mainloop()


if __name__ == "__main__":
    main()
