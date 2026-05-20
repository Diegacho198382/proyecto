import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class AlgoritmoGeneticoMejorado:
    def __init__(self, tam_poblacion=150, tam_cromosoma=20, 
                 prob_cruce=0.85, prob_mutacion=0.1, generaciones=50):
        
        self.tam_poblacion = tam_poblacion
        self.tam_cromosoma = tam_cromosoma
        self.prob_cruce = prob_cruce
        self.prob_mutacion_inicial = prob_mutacion
        self.generaciones = generaciones
        self.poblacion = []
        self.mejores_fitness = []
        
    def crear_individuo(self):
        """Crea un individuo (secuencia binaria)"""
        return [random.randint(0, 1) for _ in range(self.tam_cromosoma)]
    
    def fitness(self, individuo):
        """Función de aptitud de ejemplo: maximizar número de 1s (One-Max)"""
        return sum(individuo)  # Puedes cambiar esta función según tu problema
    
    def validar_integridad(self, individuo):
        """Validación de integridad de la secuencia"""
        return len(individuo) == self.tam_cromosoma and all(x in [0,1] for x in individuo)
    
    def inicializar_poblacion(self):
        self.poblacion = [self.crear_individuo() for _ in range(self.tam_poblacion)]
    
    def seleccion_torneo(self, k=5):
        """Selección por torneo"""
        seleccionados = []
        for _ in range(self.tam_poblacion):
            participantes = random.sample(self.poblacion, k)
            mejor = max(participantes, key=self.fitness)
            seleccionados.append(deepcopy(mejor))
        return seleccionados
    
    def cruce(self, padre1, padre2):
        """Cruce de dos puntos o uniforme"""
        if random.random() < 0.7:  # Cruce de dos puntos
            pt1 = random.randint(1, self.tam_cromosoma-2)
            pt2 = random.randint(pt1, self.tam_cromosoma-1)
            hijo1 = padre1[:pt1] + padre2[pt1:pt2] + padre1[pt2:]
            hijo2 = padre2[:pt1] + padre1[pt1:pt2] + padre2[pt2:]
        else:  # Cruce uniforme
            hijo1 = [padre1[i] if random.random() < 0.5 else padre2[i] for i in range(self.tam_cromosoma)]
            hijo2 = [padre2[i] if random.random() < 0.5 else padre1[i] for i in range(self.tam_cromosoma)]
        return hijo1, hijo2
    
    def mutacion_adaptativa(self, individuo, generacion_actual):
        """Mutación adaptativa"""
        prob_mut = self.prob_mutacion_inicial * (1 - generacion_actual / self.generaciones)
        for i in range(self.tam_cromosoma):
            if random.random() < prob_mut:
                individuo[i] = 1 - individuo[i]
        return individuo
    
    def ejecutar(self):
        self.inicializar_poblacion()
        mejor_global = None
        fitness_mejor_global = -float('inf')
        
        for gen in range(self.generaciones):
            # Evaluar fitness
            fitness_pob = [self.fitness(ind) for ind in self.poblacion]
            
            # Mejor actual
            mejor_idx = np.argmax(fitness_pob)
            mejor_actual = self.poblacion[mejor_idx]
            fitness_actual = fitness_pob[mejor_idx]
            
            if fitness_actual > fitness_mejor_global:
                mejor_global = deepcopy(mejor_actual)
                fitness_mejor_global = fitness_actual
            
            self.mejores_fitness.append(fitness_mejor_global)
            
            # Elitismo (mejor siempre pasa)
            nueva_poblacion = [deepcopy(mejor_actual)]
            
            # Selección + Cruce + Mutación
            seleccionados = self.seleccion_torneo(k=5)
            
            for i in range(0, self.tam_poblacion-1, 2):
                padre1 = seleccionados[i]
                padre2 = seleccionados[i+1]
                
                if random.random() < self.prob_cruce:
                    hijo1, hijo2 = self.cruce(padre1, padre2)
                else:
                    hijo1, hijo2 = deepcopy(padre1), deepcopy(padre2)
                
                hijo1 = self.mutacion_adaptativa(hijo1, gen)
                hijo2 = self.mutacion_adaptativa(hijo2, gen)
                
                # Validación de integridad
                if self.validar_integridad(hijo1):
                    nueva_poblacion.append(hijo1)
                if self.validar_integridad(hijo2) and len(nueva_poblacion) < self.tam_poblacion:
                    nueva_poblacion.append(hijo2)
            
            self.poblacion = nueva_poblacion[:self.tam_poblacion]
        
        return mejor_global, fitness_mejor_global

# ===================== USO =====================
if __name__ == "__main__":
    ga = AlgoritmoGeneticoMejorado(tam_poblacion=150, generaciones=80)
    mejor, fitness = ga.ejecutar()
    
    print(f"Mejor solución encontrada: {mejor}")
    print(f"Fitness final: {fitness}")
    
    # Gráfica de comparación
    plt.figure(figsize=(10,6))
    plt.plot(ga.mejores_fitness, label='Algoritmo Mejorado', color='red', linewidth=2)
    plt.title('Evolución del Fitness - Algoritmo Genético Mejorado')
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()