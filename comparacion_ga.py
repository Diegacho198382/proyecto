import random
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

# ===================== ALGORITMO ORIGINAL =====================
class AlgoritmoGeneticoOriginal:
    def __init__(self, tam_poblacion=150, tam_cromosoma=20, 
                 prob_cruce=0.8, prob_mutacion=0.1, generaciones=60):
        self.tam_poblacion = tam_poblacion
        self.tam_cromosoma = tam_cromosoma
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.generaciones = generaciones
        self.mejores_fitness = []
    
    def crear_individuo(self):
        return [random.randint(0, 1) for _ in range(self.tam_cromosoma)]
    
    def fitness(self, individuo):
        return sum(individuo)
    
    def seleccion_ruleta(self, poblacion):
        total = sum(self.fitness(ind) for ind in poblacion)
        if total == 0: return deepcopy(random.choice(poblacion))
        pick = random.uniform(0, total)
        current = 0
        for ind in poblacion:
            current += self.fitness(ind)
            if current >= pick:
                return deepcopy(ind)
    
    def cruce(self, p1, p2):
        if random.random() > self.prob_cruce:
            return deepcopy(p1), deepcopy(p2)
        punto = random.randint(1, self.tam_cromosoma-1)
        return p1[:punto] + p2[punto:], p2[:punto] + p1[punto:]
    
    def mutacion(self, individuo):
        for i in range(self.tam_cromosoma):
            if random.random() < self.prob_mutacion:
                individuo[i] = 1 - individuo[i]
        return individuo
    
    def ejecutar(self):
        poblacion = [self.crear_individuo() for _ in range(self.tam_poblacion)]
        mejor_global = -float('inf')
        
        for gen in range(self.generaciones):
            fitness_pob = [self.fitness(ind) for ind in poblacion]
            mejor_actual = max(fitness_pob)
            if mejor_actual > mejor_global:
                mejor_global = mejor_actual
            self.mejores_fitness.append(mejor_global)
            
            nueva_poblacion = []
            for _ in range(self.tam_poblacion // 2):
                p1 = self.seleccion_ruleta(poblacion)
                p2 = self.seleccion_ruleta(poblacion)
                h1, h2 = self.cruce(p1, p2)
                nueva_poblacion.extend([self.mutacion(h1), self.mutacion(h2)])
            poblacion = nueva_poblacion[:self.tam_poblacion]
        return self.mejores_fitness


# ===================== ALGORITMO MEJORADO =====================
class AlgoritmoGeneticoMejorado:
    def __init__(self, tam_poblacion=150, tam_cromosoma=20, 
                 prob_cruce=0.85, prob_mutacion=0.12, generaciones=60):
        self.tam_poblacion = tam_poblacion
        self.tam_cromosoma = tam_cromosoma
        self.prob_cruce = prob_cruce
        self.prob_mutacion_inicial = prob_mutacion
        self.generaciones = generaciones
        self.mejores_fitness = []
    
    def crear_individuo(self):
        return [random.randint(0, 1) for _ in range(self.tam_cromosoma)]
    
    def fitness(self, individuo):
        return sum(individuo)
    
    def seleccion_torneo(self, poblacion, k=5):
        seleccionados = []
        for _ in range(self.tam_poblacion):
            candidatos = random.sample(poblacion, k)
            seleccionados.append(deepcopy(max(candidatos, key=self.fitness)))
        return seleccionados
    
    def cruce(self, p1, p2):
        if random.random() < 0.6:  # Cruce de dos puntos
            pt1 = random.randint(1, self.tam_cromosoma-2)
            pt2 = random.randint(pt1+1, self.tam_cromosoma-1)
            h1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
            h2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
        else:  # Cruce uniforme
            h1 = [p1[i] if random.random() < 0.5 else p2[i] for i in range(self.tam_cromosoma)]
            h2 = [p2[i] if random.random() < 0.5 else p1[i] for i in range(self.tam_cromosoma)]
        return h1, h2
    
    def mutacion_adaptativa(self, individuo, gen):
        prob = self.prob_mutacion_inicial * (1 - gen / self.generaciones)
        for i in range(self.tam_cromosoma):
            if random.random() < prob:
                individuo[i] = 1 - individuo[i]
        return individuo
    
    def ejecutar(self):
        poblacion = [self.crear_individuo() for _ in range(self.tam_poblacion)]
        mejor_global = -float('inf')
        
        for gen in range(self.generaciones):
            fitness_pob = [self.fitness(ind) for ind in poblacion]
            mejor_actual = max(fitness_pob)
            if mejor_actual > mejor_global:
                mejor_global = mejor_actual
            self.mejores_fitness.append(mejor_global)
            
            # Elitismo
            elite = deepcopy(max(poblacion, key=self.fitness))
            nueva_poblacion = [elite]
            
            seleccionados = self.seleccion_torneo(poblacion)
            for i in range(0, self.tam_poblacion-1, 2):
                h1, h2 = self.cruce(seleccionados[i], seleccionados[i+1])
                h1 = self.mutacion_adaptativa(h1, gen)
                h2 = self.mutacion_adaptativa(h2, gen)
                nueva_poblacion.extend([h1, h2])
            
            poblacion = nueva_poblacion[:self.tam_poblacion]
        return self.mejores_fitness


# ===================== COMPARACIÓN =====================
if __name__ == "__main__":
    random.seed(42)  # Para reproducibilidad
    
    print("Ejecutando Algoritmo Original...")
    ga_orig = AlgoritmoGeneticoOriginal()
    fitness_original = ga_orig.ejecutar()
    
    print("Ejecutando Algoritmo Mejorado...")
    ga_mej = AlgoritmoGeneticoMejorado()
    fitness_mejorado = ga_mej.ejecutar()
    
    # Gráfica Comparativa
    plt.figure(figsize=(12, 7))
    plt.plot(fitness_original, label='Algoritmo Original', color='blue', linewidth=2)
    plt.plot(fitness_mejorado, label='Algoritmo Mejorado', color='red', linewidth=2.5)
    
    plt.title('Comparación de Rendimiento: Algoritmo Genético Original vs Mejorado', fontsize=14, fontweight='bold')
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Mejor Fitness', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f"\nFitness Final - Original : {fitness_original[-1]:.2f}")
    print(f"Fitness Final - Mejorado : {fitness_mejorado[-1]:.2f}")