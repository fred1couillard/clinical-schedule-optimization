import streamlit as st
import pandas as pd
from docplex.mp.model import Model

st.set_page_config(page_title="Optimisation Clinique (Goal Programming)", layout="wide")
st.title("Générateur d'Horaires - Modèle Clinique")
st.markdown("Basé sur la Programmation par Objectifs Multiples (Goal Programming).")

with st.sidebar:
    st.header("Paramètres de volume")
    cible_examens = st.slider("Cible d'examens (Nb clients)", min_value=0, max_value=50, value=20, step=1)
    cible_soins = st.slider("Cible de soins (Blocs de 1 heure)", min_value=0, max_value=400, value=150, step=5)
    
    st.divider()
    st.markdown("**Bassin d'employés :**\n- 5 Techniciens (40h/semaine garanties)\n- 5 Spécialistes (Lun au Mer uniquement)\n- 10 Auxiliaires (Min 2 jours/semaine)")

if st.button("Générer l'horaire optimal", type="primary"):
    
    with st.spinner("Veuillez patienter... Résolution du modèle (Max 60 sec)"):
        
        mdl = Model('Clinique_Optimisation_GP')

        # Ensembles (Sets)
        Techs = range(5)
        Specs = range(5)
        Auxs = range(10)
        Jours = range(7)
        Periodes = range(52)
        NbSalles = 5 

        # Variables de décision (Tâches)
        StartExam = mdl.binary_var_cube(Techs, Jours, Periodes, name='Exam')
        StartAna = mdl.binary_var_cube(Techs, Jours, Periodes, name='Ana')
        StartSuivi = mdl.binary_var_cube(Techs, Jours, Periodes, name='Suivi')
        StartEval = mdl.binary_var_cube(Specs, Jours, Periodes, name='Eval')
        StartSoin_Tech = mdl.binary_var_cube(Techs, Jours, Periodes, name='Soin_T')
        StartSoin_Aux = mdl.binary_var_cube(Auxs, Jours, Periodes, name='Soin_A')
        
        TechDiner = mdl.binary_var_cube(Techs, Jours, Periodes, name='TDin')
        AuxDiner = mdl.binary_var_cube(Auxs, Jours, Periodes, name='ADin')

        # Variables de décision (Quarts de travail)
        TechTrav = mdl.binary_var_matrix(Techs, Jours, name='TTrav') 
        SpecTrav = mdl.binary_var_matrix(Specs, Jours, name='STrav')
        AuxS1 = mdl.binary_var_matrix(Auxs, Jours, name='AS1') 
        AuxS2 = mdl.binary_var_matrix(Auxs, Jours, name='AS2') 
        AuxS3 = mdl.binary_var_matrix(Auxs, Jours, name='AS3') 
        AuxTrav = mdl.binary_var_matrix(Auxs, Jours, name='ATrav')

        TechActif = mdl.binary_var_list(Techs, name='TActif')
        SpecActif = mdl.binary_var_list(Specs, name='SActif')
        AuxActif = mdl.binary_var_list(Auxs, name='AActif')

        # Variables de déviations (Goal Programming)
        d_plus = mdl.continuous_var_dict(['Shifts', 'SoinTech'], name='d_plus')
        d_moins = mdl.continuous_var_dict(['Shifts', 'SoinTech'], name='d_moins')

        # Variables d'occupation et transition
        TechBusy = mdl.binary_var_cube(Techs, Jours, Periodes, name='TBusy')
        SpecBusy = mdl.binary_var_cube(Specs, Jours, Periodes, name='SBusy')
        AuxBusy = mdl.binary_var_cube(Auxs, Jours, Periodes, name='ABusy')

        TechComm = mdl.continuous_var_cube(Techs, Jours, Periodes, lb=0, ub=1)
        AuxComm = mdl.continuous_var_cube(Auxs, Jours, Periodes, lb=0, ub=1)

        # 1. Lier tâches et occupations
        for d in Jours:
            for p in Periodes:
                for i in Techs:
                    occup = mdl.sum(StartExam[i,d,p-k] for k in range(4) if p-k >= 0) + \
                            mdl.sum(StartAna[i,d,p-k] for k in range(4) if p-k >= 0) + \
                            mdl.sum(StartSoin_Tech[i,d,p-k] for k in range(4) if p-k >= 0) + \
                            mdl.sum(StartSuivi[i,d,p-k] for k in range(2) if p-k >= 0) + \
                            mdl.sum(TechDiner[i,d,p-k] for k in range(2) if p-k >= 0)
                    mdl.add_constraint(TechBusy[i,d,p] == occup)
                    mdl.add_constraint(TechBusy[i,d,p] <= TechTrav[i,d])

                for j in Specs:
                    occup = mdl.sum(StartEval[j,d,p-k] for k in range(2) if p-k >= 0)
                    mdl.add_constraint(SpecBusy[j,d,p] == occup)
                    mdl.add_constraint(SpecBusy[j,d,p] <= SpecTrav[j,d])

                for e in Auxs:
                    occup = mdl.sum(StartSoin_Aux[e,d,p-k] for k in range(4) if p-k >= 0) + \
                            mdl.sum(AuxDiner[e,d,p-k] for k in range(2) if p-k >= 0)
                    mdl.add_constraint(AuxBusy[e,d,p] == occup)

        # 2. Quarts et pauses repas
        for d in Jours:
            for i in Techs:
                if d in [5, 6]: mdl.add_constraint(TechTrav[i,d] == 0) 
                mdl.add_constraint(mdl.sum(TechDiner[i,d,p] for p in Periodes) == TechTrav[i,d])
                for p in Periodes:
                    if p >= 32: mdl.add_constraint(TechBusy[i,d,p] == 0) 
                    if not (14 <= p <= 20): mdl.add_constraint(TechDiner[i,d,p] == 0)

            for e in Auxs:
                mdl.add_constraint(AuxS1[e,d] + AuxS2[e,d] + AuxS3[e,d] <= 1)
                mdl.add_constraint(AuxTrav[e,d] == AuxS1[e,d] + AuxS2[e,d] + AuxS3[e,d])
                mdl.add_constraint(mdl.sum(AuxDiner[e,d,p] for p in Periodes) == AuxS1[e,d] + AuxS2[e,d])
                for p in Periodes:
                    en_shift = 0
                    if 0 <= p <= 31: en_shift += AuxS1[e,d]
                    if 18 <= p <= 51: en_shift += AuxS2[e,d]
                    if 32 <= p <= 51: en_shift += AuxS3[e,d]
                    mdl.add_constraint(AuxBusy[e,d,p] <= en_shift)

                    if 14 <= p <= 20: mdl.add_constraint(AuxDiner[e,d,p] <= AuxS1[e,d])
                    elif 32 <= p <= 38: mdl.add_constraint(AuxDiner[e,d,p] <= AuxS2[e,d])
                    else: mdl.add_constraint(AuxDiner[e,d,p] == 0)

        # 3. Capacité de salles 
        for d in Jours:
            for p in Periodes:
                salles_occupees = 0
                
                # Pour les techniciens
                if 0 <= p <= 31:
                    salles_occupees += mdl.sum(TechTrav[i,d] for i in Techs)
                    
                # Pour les auxiliaires (dépend du quart)
                aux_occup = 0
                if 0 <= p <= 31:
                    aux_occup += mdl.sum(AuxS1[e,d] for e in Auxs)
                if 18 <= p <= 51:
                    aux_occup += mdl.sum(AuxS2[e,d] for e in Auxs)
                if 32 <= p <= 51:
                    aux_occup += mdl.sum(AuxS3[e,d] for e in Auxs)
                    
                salles_occupees += aux_occup
                mdl.add_constraint(salles_occupees <= NbSalles)

        # 4. Couvertures et lissage
        for d in range(5): 
            mdl.add_constraint(mdl.sum(TechTrav[i,d] for i in Techs) + mdl.sum(AuxS1[e,d] for e in Auxs) >= 1)
            mdl.add_constraint(mdl.sum(AuxS2[e,d] for e in Auxs) + mdl.sum(AuxS3[e,d] for e in Auxs) >= 1)
            
            # Limite d'examens par jour pour éviter un goulot d'étranglement
            mdl.add_constraint(mdl.sum(StartExam[i,d,p] for i in Techs for p in Periodes) <= 8)
            
        for d in [5, 6]:
            mdl.add_constraint(mdl.sum(AuxS1[e,d] for e in Auxs) >= 1)
            mdl.add_constraint(mdl.sum(AuxS2[e,d] for e in Auxs) == 0)
            mdl.add_constraint(mdl.sum(AuxS3[e,d] for e in Auxs) == 0)

        # 5. Cibles de production
        mdl.add_constraint(mdl.sum(StartExam[i,d,p] for i in Techs for d in Jours for p in Periodes) == cible_examens)
        mdl.add_constraint(mdl.sum(StartAna[i,d,p] for i in Techs for d in Jours for p in Periodes) == cible_examens)
        mdl.add_constraint(mdl.sum(StartSuivi[i,d,p] for i in Techs for d in Jours for p in Periodes) == cible_examens)
        mdl.add_constraint(mdl.sum(StartSoin_Aux[e,d,p] for e in Auxs for d in Jours for p in Periodes) + 
                           mdl.sum(StartSoin_Tech[i,d,p] for i in Techs for d in Jours for p in Periodes) == cible_soins)

        # 6. Ordonnancement logique
        for d in Jours:
            for p in range(3): mdl.add_constraint(mdl.sum(StartEval[j,d,p] for j in Specs) == 0)
            for p in range(3, 52):
                mdl.add_constraint(mdl.sum(StartEval[j,d,p] for j in Specs) == mdl.sum(StartExam[i,d,p-3] for i in Techs))
        
        # 7. Contraintes horaires spécifiques
        for d in Jours:
            for p in Periodes:
                # Fin de journée
                if p >= 49: 
                    mdl.add_constraint(mdl.sum(StartExam[i,d,p] for i in Techs) == 0)
                    mdl.add_constraint(mdl.sum(StartAna[i,d,p] for i in Techs) == 0)
                    mdl.add_constraint(mdl.sum(StartSoin_Tech[i,d,p] for i in Techs) == 0)
                    mdl.add_constraint(mdl.sum(StartSoin_Aux[e,d,p] for e in Auxs) == 0)
                if p >= 51:
                    mdl.add_constraint(mdl.sum(StartSuivi[i,d,p] for i in Techs) == 0)
                    mdl.add_constraint(mdl.sum(StartEval[j,d,p] for j in Specs) == 0)
                
                # Matin vs Après-midi
                if d not in [0, 1, 2] or p > 15:
                    mdl.add_constraint(mdl.sum(StartExam[i,d,p] for i in Techs) == 0)
                    mdl.add_constraint(mdl.sum(StartEval[j,d,p] for j in Specs) == 0)
                
                if d in [0, 1, 2] and p <= 15:
                    mdl.add_constraint(mdl.sum(StartAna[i,d,p] for i in Techs) == 0)
                    mdl.add_constraint(mdl.sum(StartSuivi[i,d,p] for i in Techs) == 0)

                # Transitions
                for i in Techs:
                    if p == 0: mdl.add_constraint(TechComm[i,d,p] >= TechBusy[i,d,p])
                    else: mdl.add_constraint(TechComm[i,d,p] >= TechBusy[i,d,p] - TechBusy[i,d,p-1])
                for e in Auxs:
                    if p == 0: mdl.add_constraint(AuxComm[e,d,p] >= AuxBusy[e,d,p])
                    else: mdl.add_constraint(AuxComm[e,d,p] >= AuxBusy[e,d,p] - AuxBusy[e,d,p-1])
            
            for i in Techs: mdl.add_constraint(mdl.sum(TechComm[i,d,p] for p in Periodes) <= 2)
            for e in Auxs: mdl.add_constraint(mdl.sum(AuxComm[e,d,p] for p in Periodes) <= 2)

        # 8. Contrats de travail
        for i in Techs:
            mdl.add_constraint(mdl.sum(TechTrav[i,d] for d in Jours) == 5 * TechActif[i])
        for e in Auxs:
            mdl.add_constraint(mdl.sum(AuxTrav[e,d] for d in Jours) <= 7 * AuxActif[e])
            mdl.add_constraint(mdl.sum(AuxTrav[e,d] for d in Jours) >= 2 * AuxActif[e])
        for j in Specs:
            mdl.add_constraint(mdl.sum(SpecTrav[j,d] for d in Jours) <= 3 * SpecActif[j])
            mdl.add_constraint(mdl.sum(SpecTrav[j,d] for d in Jours) >= 1 * SpecActif[j])
            # Les spécialistes ne travaillent pas de Jeudi à Dimanche
            for d in [3, 4, 5, 6]:
                mdl.add_constraint(SpecTrav[j,d] == 0)

        # Fonction objectif (Goal Programming)
        cout_shifts = mdl.sum(TechTrav[i,d] * 10000 for i in Techs for d in Jours) + \
                      mdl.sum(SpecTrav[j,d] * 10000 for j in Specs for d in Jours) + \
                      mdl.sum(AuxTrav[e,d] * 5000 for e in Auxs for d in Jours)
        
        cout_soin_tech = mdl.sum(StartSoin_Tech[i,d,p] * 50 for i in Techs for d in Jours for p in Periodes)
        
        mdl.add_constraint(cout_shifts - d_plus['Shifts'] + d_moins['Shifts'] == 0)
        mdl.add_constraint(cout_soin_tech - d_plus['SoinTech'] + d_moins['SoinTech'] == 0)
        
        mdl.minimize(d_plus['Shifts'] + d_plus['SoinTech'])

        # Résolution
        mdl.set_time_limit(60) 
        mdl.parameters.randomseed = 12345 
        
        solution = mdl.solve(log_output=False)

        if solution:
            st.success(f"Horaire optimisé ! Coût total des déviations : {int(solution.objective_value)} $")
            
            jours_noms = ["1-Lundi", "2-Mardi", "3-Mercredi", "4-Jeudi", "5-Vendredi", "6-Samedi", "7-Dimanche"]
            def get_time_str(p): return f"{8 + p // 4:02d}h{(p % 4) * 15:02d}"

            records = []
            
            for d in Jours:
                for p in Periodes:
                    t_str = get_time_str(p)
                    
                    for i in Techs:
                        if solution.get_value(TechActif[i]) > 0.5:
                            tache = ""
                            if solution.get_value(StartExam[i,d,p]) > 0.5: tache = "🟢 Examen"
                            elif any(solution.get_value(StartExam[i,d,p-k]) > 0.5 for k in range(1, 4) if p-k >= 0): tache = " ⬇ "
                            elif solution.get_value(StartAna[i,d,p]) > 0.5: tache = "🔵 Analyse"
                            elif any(solution.get_value(StartAna[i,d,p-k]) > 0.5 for k in range(1, 4) if p-k >= 0): tache = " ⬇ "
                            elif solution.get_value(StartSuivi[i,d,p]) > 0.5: tache = "🟡 Suivi"
                            elif any(solution.get_value(StartSuivi[i,d,p-k]) > 0.5 for k in range(1, 2) if p-k >= 0): tache = " ⬇ "
                            elif solution.get_value(StartSoin_Tech[i,d,p]) > 0.5: tache = "🟣 Soin"
                            elif any(solution.get_value(StartSoin_Tech[i,d,p-k]) > 0.5 for k in range(1, 4) if p-k >= 0): tache = " ⬇ "
                            elif solution.get_value(TechDiner[i,d,p]) > 0.5: tache = "PAUSE DÎNER"
                            elif any(solution.get_value(TechDiner[i,d,p-k]) > 0.5 for k in range(1, 2) if p-k >= 0): tache = " ⬇ "
                            elif solution.get_value(TechTrav[i,d]) > 0.5 and 0 <= p <= 31: tache = "Admin / Libre"
                            if tache: records.append({"Jour": jours_noms[d], "Heure": t_str, "Employé": f"Tech {i+1}", "Tâche": tache})

                    for j in Specs:
                        if solution.get_value(SpecActif[j]) > 0.5:
                            tache = ""
                            if solution.get_value(StartEval[j,d,p]) > 0.5: tache = "🔴 Évaluation"
                            elif any(solution.get_value(StartEval[j,d,p-k]) > 0.5 for k in range(1, 2) if p-k >= 0): tache = " ⬇ "
                            elif solution.get_value(SpecTrav[j,d]) > 0.5: tache = "Dispo / Admin"
                            if tache: records.append({"Jour": jours_noms[d], "Heure": t_str, "Employé": f"Spécialiste {j+1}", "Tâche": tache})
                            
                    for e in Auxs:
                        if solution.get_value(AuxActif[e]) > 0.5:
                            tache = ""
                            if solution.get_value(StartSoin_Aux[e,d,p]) > 0.5: tache = "🟣 Soin"
                            elif any(solution.get_value(StartSoin_Aux[e,d,p-k]) > 0.5 for k in range(1, 4) if p-k >= 0): tache = " ⬇ "
                            elif solution.get_value(AuxDiner[e,d,p]) > 0.5: tache = "PAUSE REPAS"
                            elif any(solution.get_value(AuxDiner[e,d,p-k]) > 0.5 for k in range(1, 2) if p-k >= 0): tache = " ⬇ "
                            elif solution.get_value(AuxS1[e,d]) > 0.5 and 0 <= p <= 31: tache = "Dispo (Quart 8h-16h)"
                            elif solution.get_value(AuxS2[e,d]) > 0.5 and 18 <= p <= 51: tache = "Dispo (Quart 12h30-21h)"
                            elif solution.get_value(AuxS3[e,d]) > 0.5 and 32 <= p <= 51: tache = "Dispo (Quart 16h-21h)"
                            if tache: records.append({"Jour": jours_noms[d], "Heure": t_str, "Employé": f"Auxiliaire {e+1}", "Tâche": tache})

            if records:
                df = pd.DataFrame(records)
                jours_actifs = sorted(df['Jour'].unique())
                onglets = st.tabs([j.split('-')[1] for j in jours_actifs])
                
                for index, jour in enumerate(jours_actifs):
                    with onglets[index]:
                        df_jour = df[df['Jour'] == jour].drop(columns=['Jour'])
                        df_pivot = df_jour.pivot_table(index="Heure", columns="Employé", values="Tâche", aggfunc='first').fillna("")
                        st.dataframe(df_pivot, height=800, use_container_width=True)
            else:
                st.info("L'horaire est vide.")
        else:
            st.error("Impossible de générer l'horaire. Réduisez légèrement vos cibles ou ajustez les contraintes.")
