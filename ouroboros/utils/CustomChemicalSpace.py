import random
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.rdChemReactions import PreprocessReaction

custom_rxn_list = [
    "[*:101]-[#6]=[#8].[#7+]-[*:102]>>[#6]1(=[#6](-[#7]=[#6]2-[#7]-1-[#6]=[#6]-[#6]=[#6]-2)-[*:101])-[#7]-[*:102]",
    "[*:101]-[#6]-[#6](=[#8])-[*:102].[#7+]-[*:103]>>[#16]1-[#6](=[#6](-[#6](=[#6]-1-[*:101])-[*:102])-[#6](-[#7](-[*:103])-[H])=[#8])-[#7]",
    "[*:101]-[#6]-[#6](=[#8])-[*:102].[#7+]-[*:103].[*:104]-[#6](=[#8])-[#17]>>[#16]1-[#6](=[#6](-[#6](=[#6]-1-[*:101])-[*:102])-[#6](-[#7](-[*:103])-[H])=[#8])-[#7]-[#6](=[#8])-[*:104]",
    "[*:101]-[#6](-[#8-])=[#8].[#6](-[*:103])(=[#8])-[*:102].[#7+]-[*:104]>>[*:101]-[#6](-[#8]-[#6](-[#6](=[#8])-[#7]-[*:104])(-[*:102])-[*:103])=[#8]",
    "[*:101]-[#6]=[#8].[*:102]-[#6](-[#8-])=[#8].[#7+]-[*:103]>>[#6](=[#8])(-[*:102])-[#7]1-[#6]-[#6]-[#7](-[#6]-[#6]-1)-[#6](-[*:101])-[#6](-[#7]-[*:103])=[#8]",
    "[*:101]-[#6](-[*:102])=[#8].[#7+]-[*:103]>>[#6]1(=[#7]-[#7]=[#7]-[#7]-1-[*:103])-[#6](-[*:102])(-[*:101])-[#8]",
    "[*:102]-[#6]=[#8].[*:101]-[#7+].[*:103]-[#7+]>>[#7](-[*:103])-[#6](-[#6](-[#7]-[*:101])-[*:102])=[#8]",
    "[*:101]-[#7+].[*:102]-[#6](-[*:103])=[#8].[*:104]-[#6](=[#8])-[#8-].[#7+]-[*:105]>>[#6](=[#8])(-[#7]-[*:105])-[#6](-[*:102])(-[*:103])-[#7](-[*:101])-[#6](=[#8])-[*:104]",
    "[*:101]-[#7+].[#6](-[#8-])(=[#8])-[*:102].[*:103]-[#6](-[#6]=[#8])=[#8].[#7+]-[*:104]>>[#6]1(-[#7](-[#6](=[#6](-[#7]=1)-[*:103])-[#6](-[#7]-[*:104])=[#8])-[*:101])-[*:102]",
    "[*:101]-[#7+]-[*:102].[*:104]-[#6](-[*:103])=[#8].[#7+]-[*:105]>>[#7](-[#6](-[#6]1=[#7]-[#7]=[#7]-[#7]-1-[*:105])(-[*:103])-[*:104])(-[*:101])-[*:102]"
]

class CustomChemicalSpace():
    def __init__(
            self, 
            rxn_formula = '',
            reactants = []
        ):
        self.Reaction = rdChemReactions.ReactionFromSmarts(rxn_formula)
        self.Reaction.Initialize()
        nWarn, nError, nReacts, nProds, reactantLabels = PreprocessReaction(self.Reaction)
        print(f'NOTE: Loading {rxn_formula}....')
        if nError > 0 or nWarn > 0:
            raise ValueError(f'Warning: {nWarn}, Error: {nError}')
        else:
            print(f'NOTE: Reactant number: {nReacts}, Product Number: {nProds}')
        assert nWarn + nError == 0
        assert nReacts > 0
        assert nProds == 1
        self.ReactantTemplate = {
            reactant_id: self.Reaction.GetReactantTemplate(reactant_id)
            for reactant_id in range(nReacts)
        }
        self.reactants_library = {
            reactant_id: {
                'SMILES': [],
                'MOL': []
            }
            for reactant_id in range(nReacts)
        }
        for reactant in reactants:
            mol = Chem.MolFromSmiles(reactant)
            for react_id, template in self.ReactantTemplate.items():
                if mol.HasSubstructMatch(template):
                    self.reactants_library[react_id]['SMILES'].append(reactant)
                    self.reactants_library[react_id]['MOL'].append(mol)
        for reactant_id in self.reactants_library:
            print(f'NOTE: Reactant {reactant_id}: {len(self.reactants_library[reactant_id]["SMILES"])}')

    def reaction(self, mol_set):
        return self.Reaction.RunReactants(mol_set)[0][0]
        
    def __call__(self, num_samples=10):
        sampled_products = []
        while len(sampled_products) < num_samples:
            reactant_mols = []
            for reactant_id in self.reactants_library:
                if self.reactants_library[reactant_id]['MOL']:
                    reactant_mols.append(random.choice(self.reactants_library[reactant_id]['MOL']))
            if len(reactant_mols) == len(self.reactants_library):
                try:
                    product = self.reaction(tuple(reactant_mols))
                    if product:
                        Chem.SanitizeMol(product)
                        sampled_products.append(Chem.MolToSmiles(product))
                except:
                    continue
        return sampled_products  
    



