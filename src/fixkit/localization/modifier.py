from abc import ABC
import abc
from typing import List
from fixkit.localization.location import WeightedIdentifier

class LocationModifier(ABC):
    """
    Modifies the suggestions from localization to only use certain locations 
    or apply a specific mutation chance.
    """
    @abc.abstractmethod
    def locations(self, suggestions: List[WeightedIdentifier]) -> List[WeightedIdentifier]:
        pass

    @abc.abstractmethod
    def mutation_chance(self, location: WeightedIdentifier) -> bool:
        pass


class DefaultModifier(LocationModifier):
    """
    Does not change the original implementation of genetic repair.
    All suggestions are used for mutations with the weights as the mutation chance.
    """
    def __init__(self):
        pass

    def locations(self, suggestions: List[WeightedIdentifier]) -> List[WeightedIdentifier]:
        return suggestions[:]
    
    def mutation_chance(self, location: WeightedIdentifier) -> bool:
        return location.weight
    

class TopRankModifier(LocationModifier):
    """
    Only considers the first "top_k" locations to mutate.
    Applies an equal mutation chance of 1.0 to all locations.
    """
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def locations(self, suggestions: List[WeightedIdentifier]) -> List[WeightedIdentifier]:
        return [location for location in suggestions[:self.top_k] if location.weight > 0]
    
    def mutation_chance(self, location: WeightedIdentifier) -> bool:
        return 1.0
    
class TopEqualRankModifier(LocationModifier):
    """
    Only considers the locations to mutate, which are under the first "top_k" weights
    and above the fitness threshold. Applies an equal mutation chance of 1.0 to all locations.
    """
    def __init__(self, top_k: int = 3, threshold: float = 0.0):
        self.top_k = top_k
        self.threshold = threshold

    def locations(self, suggestions: List[WeightedIdentifier]) -> List[WeightedIdentifier]:
        top_weights = []
        for suggestion in suggestions:
            if suggestion.weight not in top_weights:
                top_weights.append(suggestion.weight)
            if len(top_weights) >= self.top_k:
                break

        return [location for location in suggestions 
                if location.weight in top_weights and location.weight > self.threshold]
    
    def mutation_chance(self, location: WeightedIdentifier) -> float:
        return 1.0
    

class SigmoidModifier(LocationModifier):
    """
    Uses a sigmoid curve with a specified "steepness" and "midpoint"
    to alter the weight. Values above the midpoint are pushed closer to 1.0,
    while values below the midpoint are pushed to 0.0.
    """
    def __init__(self, steepness: int = 10, midpoint: int = 0.8):
        self.steepness = steepness
        self.midpoint = midpoint

    def locations(self, suggestions: List[WeightedIdentifier]) -> List[WeightedIdentifier]:
        return [
            WeightedIdentifier(location.identifier, self._sigmoid_function(location.weight)) 
            for location in suggestions
        ]
    
    def _sigmoid_function(self, weight: float) -> float:
        x = weight
        a = self.steepness
        m = self.midpoint
        return ((x / m) ** a) / ((x / m) ** a + ((1 - x) / (1 - m)) ** a)
    
    def mutation_chance(self, location: WeightedIdentifier) -> float:
        return location.weight
    