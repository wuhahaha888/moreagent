"""Original authored MedQA-style extensions. EVALUATOR ONLY: contains answers.

Synthetic patients; factual answer concepts checked against the source catalog.
Difficulty is a design target, not a measured error rate or clinical certification.
"""


def item(code, group, question, correct, distractors, parents, reuse, gap, trap,
         rationale, sources, types):
    return {'question_id':'medqa-derived::'+code,'group_id':group,'question':question,
        'correct':correct,'distractors':[{'text':text,'why_wrong_zh':why} for text,why in distractors],
        'parents':parents,'reusable_knowledge_zh':reuse,'remaining_gap_zh':gap,
        'negative_transfer_zh':trap,'rationale_zh':rationale,'source_keys':sources,
        'continuity_types':types,'difficulty_design':'multi_step_challenge_unmeasured',
        'independent_clinical_review':'pending','model_blind_test':'not_run'}


ITEMS = [
item('A01','anemia',
'''A 58-year-old woman is evaluated for persistent fatigue and progressive difficulty walking in the dark. She has autoimmune thyroid disease and intermittently takes ibuprofen for arthritis. Hemoglobin is 8.8 g/dL, MCV 84 fL, RDW 25%, ferritin 6 ng/mL, and transferrin saturation 5%. A smear contains both hypochromic microcytes and macro-ovalocytes, with occasional hypersegmented neutrophils. Examination shows reduced vibration sense and proprioception in both feet. Serum vitamin B12 is 230 pg/mL (reference 200-900), folate is normal, and creatinine is 0.8 mg/dL. After four weeks of supervised iron replacement, hemoglobin increases to 10.1 g/dL, MCV rises to 101 fL, and the sensory findings persist. Which additional finding would best support the mechanism that was obscured by the initial average cell volume?''',
'An elevated serum methylmalonic acid concentration',[
('A normal methylmalonic acid concentration with an isolated elevation of homocysteine','更符合孤立叶酸通路异常，不能最好解释本题神经表现及被缺铁掩盖的B12缺乏。'),
('A strongly positive direct antiglobulin test for IgG','提示免疫性溶血，不能统一解释巨卵圆红细胞、低铁储备和感觉异常。'),
('An elevated hepcidin concentration with preserved marrow iron stores','可解释炎症性铁限制，但不能解释本题极低铁蛋白及持续的后索相关表现。')],
['medqa::0900','medqa::1009','medqa::0409'],
'组合缺铁的铁代谢证据与大细胞性贫血、感觉异常的鉴别；平均MCV不能替代涂片和混合机制判断。',
'临界B12结果在肾功能正常时如何由代谢标志物进一步核实。',
'不能因初始MCV正常就排除B12缺乏，也不能因补铁有效就认为所有症状均由缺铁解释。',
'明确缺铁与混合涂片使平均MCV失去单独排除价值；补铁后显露大细胞倾向且感觉异常持续。肾功能正常时，MMA升高比孤立同型半胱氨酸升高更支持B12相关代谢障碍。',
['b12_nice','b12_merck'],['multi_memory_combination','condition_recheck']),

item('A02','anemia',
'''A 61-year-old woman with active rheumatoid arthritis has worsening exertional fatigue. Her medications include methotrexate with folate supplementation and frequent naproxen. Six months ago, hemoglobin was 10.7 g/dL, MCV 82 fL, ferritin 310 ng/mL, and transferrin saturation 17%. Hemoglobin is now 8.4 g/dL, MCV 72 fL, RDW 21%, reticulocytes 0.6%, ferritin 72 ng/mL, and transferrin saturation 5%. C-reactive protein remains markedly elevated. The soluble transferrin receptor is above this laboratory's reference range and the transferrin-receptor/ferritin index is elevated. There is no biochemical evidence of hemolysis. She has recently developed intermittent black stools. Which explanation best accounts for the change in her anemia?''',
'Absolute iron deficiency from blood loss superimposed on inflammatory iron restriction',[
('Inflammatory iron sequestration alone, with unchanged total-body iron stores','炎症背景仍在，但新增失血、铁指标趋势及受体指数不支持仅沿用单一炎症机制。'),
('Methotrexate-induced isolated folate deficiency','孤立叶酸缺乏通常不能解释本题进行性小细胞化、低铁可用性及黑便。'),
('A new immune-mediated peripheral hemolytic process','缺乏溶血证据和相应网织红细胞反应；不能解释铁代谢组合。')],
['medqa::0394','medqa::0900','medqa-derived::A01'],
'复用炎症性贫血与绝对缺铁的对照，保留混合机制的可能性。',
'在持续炎症中结合时间趋势、失血及本实验室sTfR相关指标识别合并缺铁。',
'不能把旧题的高铁蛋白规则当成任何炎症病例均不存在缺铁。',
'铁蛋白可受炎症影响；本题不是靠单一阈值下结论，而是失血、MCV与铁指标变化及sTfR相关指标共同支持合并绝对缺铁。',
['iron_inflammation','iron_merck'],['partial_reuse','longitudinal_update']),

item('A03','anemia',
'''A 15-year-old boy with chronic nonimmune hemolytic anemia usually has hemoglobin near 10.5 g/dL and a reticulocyte count of 8-10%. Previous testing showed spherocytes, reduced eosin-5-maleimide binding, and a negative direct antiglobulin test. He takes folic acid regularly. One week after a febrile illness that also affected younger children in his household, he develops marked pallor and tachycardia. Hemoglobin is 5.6 g/dL, MCV 86 fL, reticulocytes 0.1%, leukocytes 5,900/mm3, and platelets 245,000/mm3. Bilirubin is lower than during his usual hemolytic exacerbations. B12 and folate concentrations are adequate, and there has been no bleeding. Which process best explains this acute deterioration?''',
'Transient infection-mediated arrest of erythroid precursor production',[
('An acceleration of splenic destruction with an appropriately increased erythropoietic response','会预期网织红细胞代偿增加；与本题从高基线跌至0.1%不符。'),
('New IgG-mediated destruction of mature erythrocytes','不能最好解释选择性的严重网织红细胞减少及既往非免疫病程。'),
('A folate-deficient megaloblastic crisis caused by chronically increased erythrocyte turnover','需与再障危象鉴别，但本题补充规律、叶酸充分及感染后的急骤变化更支持红系生成暂时受抑。')],
['medqa::0476','medqa::0810','medqa::0673','medqa-derived::A01'],
'复用遗传性球形红细胞病的非免疫溶血框架、叶酸需求及生成不足和外周破坏的区别。',
'感染后选择性红系抑制、尤其细小病毒B19相关再障危象的证据。',
'不能将慢性溶血背景下每次Hb下降都解释为溶血加重，也不能机械复用补叶酸答案。',
'关键是相对个人基线的网织红细胞骤降，而非Hb低本身。该模式符合感染相关暂时性红系停滞；球形红细胞病患者对这种生成中断尤其敏感。',
['hs_genereviews','parvo_primary'],['condition_reversal','partial_reuse']),

item('A04','anemia',
'''A 47-year-old woman with an autoimmune connective-tissue disease develops jaundice and fatigue over three weeks. Hemoglobin is 7.7 g/dL, MCV 98 fL, reticulocytes 11%, indirect bilirubin 3.4 mg/dL, and haptoglobin is undetectable. The smear shows numerous small dense erythrocytes without central pallor. She has no childhood anemia, no family history of early gallstones or splenectomy, and no recent transfusion. The direct antiglobulin test is strongly positive for IgG but negative for C3. Which mechanism most directly produces the observed erythrocyte shape in this patient?''',
'Partial removal of IgG-coated erythrocyte membrane by splenic macrophages',[
('Inherited defective linkage between the membrane skeleton and lipid bilayer','可造成相同形态，但不能最好解释本例获得性病程及IgG阳性DAT。'),
('Oxidant precipitation of hemoglobin followed by splenic removal of inclusions','更符合氧化损伤与咬痕细胞，不是本例DAT模式的主要机制。'),
('Intravascular terminal-complement lysis triggered primarily by cold-reactive IgM','与IgG阳性而C3阴性的检测组合及外周球形细胞形成机制不符。')],
['medqa::0476','medqa::0810','medqa-derived::A03'],
'复用球形细胞、胆红素、网织红细胞和DAT的联合鉴别，保留形态与病因分离。',
'温抗体介导的膜部分吞噬如何产生继发球形细胞。',
'形态相同不意味着继续采用遗传性膜缺陷的病因结论。',
'获得性溶血加IgG阳性DAT支持温抗体相关过程。脾巨噬细胞经Fc受体识别并移除部分膜，形成球形细胞。',
['warm_aiha'],['condition_reversal','negative_transfer_control']),

item('A05','anemia',
'''A 56-year-old man is admitted with severe fatigue, mild confusion, and numbness of both feet. Hemoglobin is 5.9 g/dL, MCV 91 fL, RDW 29%, platelets 68,000/mm3, and leukocytes 2,900/mm3. The smear contains schistocytes, macro-ovalocytes, and hypersegmented neutrophils. LDH is 7,800 U/L, haptoglobin is undetectable, and indirect bilirubin is 1.8 mg/dL. Reticulocytes are 0.7%, corresponding to a reticulocyte production index below 0.3. Creatinine is 0.9 mg/dL, coagulation studies are normal, and he has received no transfusions. B12 is 95 pg/mL, methylmalonic acid is markedly elevated, and ferritin is 8 ng/mL. Which mechanism best accounts for the apparent hemolysis and the disproportionately weak reticulocyte response?''',
'Ineffective megaloblastic erythropoiesis with intramedullary destruction of developing cells',[
('Platelet-rich microvascular thrombosis with a preserved compensatory marrow response','可解释血小板低和裂红细胞，但不能最好统一极低RPI、巨大LDH及明确B12相关证据。'),
('Isolated splenic sequestration caused by an inherited erythrocyte membrane defect','不能统一全血细胞减少、巨卵圆细胞及MMA升高。'),
('An isolated episode of occult blood loss with intact erythroid maturation','可以合并缺铁，但无法解释所有溶血样指标和巨幼细胞性表现。')],
['medqa::1009','medqa::0409','medqa-derived::A01','medqa-derived::A03','medqa-derived::A04'],
'组合混合贫血可掩盖MCV、MMA鉴别及RPI区分生成不足与外周破坏的证据。',
'核实严重B12缺乏的假性血栓性微血管病模式；未有对应原文时补查。',
'不能因见到裂红细胞就忽略生成不足，也不能因MCV正常排除巨幼细胞性机制。',
'本例B12和MMA证据明确，伴多系减少、极高LDH及不适当低RPI，更符合无效造血和髓内破坏。合并缺铁解释平均MCV未明显升高。本题询问机制，不构成排除真实TTP的临床处置规则。',
['pseudo_tma','b12_nice'],['multi_memory_combination','condition_reversal']),

item('A06','anemia',
'''A 19-year-old man with an established pathogenic ANK1 variant undergoes splenectomy for symptomatic chronic hemolysis. Before surgery, hemoglobin is 8.9 g/dL, reticulocytes 12%, and indirect bilirubin 3.1 mg/dL. Six months later, without transfusion, hemoglobin is 13.2 g/dL, reticulocytes 2.6%, and indirect bilirubin 0.8 mg/dL. The smear still contains spherocytes and now shows occasional Howell-Jolly bodies. The direct antiglobulin test remains negative. Which interpretation best reconciles the clinical improvement with the persistent erythrocyte morphology?''',
'Removal of the principal site of sequestration prolongs survival despite persistence of the intrinsic membrane defect',[
('Splenectomy restores the missing erythrocyte membrane protein, so the remaining abnormal cells must be transfused donor cells','脾切除不修复ANK1缺陷，本题也明确无输血。'),
('The original disorder was antibody-mediated, and splenectomy eliminated production of the pathogenic antibody','持续DAT阴性和已知致病变异不支持这一解释。'),
('Persistence of spherocytes establishes ongoing severe intravascular hemolysis despite the improved biochemical markers','形态异常不能单独推断严重溶血仍在，且本题Hb、网织红细胞和胆红素已改善。')],
['medqa::0476','medqa::0810','medqa::0673','medqa-derived::A03','medqa-derived::A04'],
'组合膜缺陷、脾清除与免疫性膜丢失的对照，区分病因、形态和疾病活动度。',
'脾切除改善细胞存活但不纠正遗传缺陷的作用边界。',
'不能把异常形态仍存在等同于治疗失败，也不能把脾切除等同于修复膜蛋白。',
'临床改善来自减少脾内滞留和清除；遗传性膜缺陷仍在。Howell-Jolly小体与脾滤除功能缺失一致，不表示新的巨幼细胞性危象。',
['hs_genereviews','hs_splenectomy'],['delayed_reuse','multi_memory_combination']),

item('A07','anemia',
'''A 64-year-old man has eight months of fatigue, distal numbness, and progressive gait instability. Hemoglobin is 8.9 g/dL, MCV 95 fL, RDW 19%, and absolute neutrophil count 800/mm3. Platelets are normal. Ferritin is 280 ng/mL and transferrin saturation 28%. B12, methylmalonic acid, and folate are normal. An eight-week supervised iron course has not improved the anemia. Bone marrow examination shows vacuolated erythroid and granulocytic precursors with scattered ring sideroblasts, no excess blasts, and a normal karyotype. He uses large amounts of a zinc-containing denture adhesive and takes additional zinc supplements. Which mechanism best explains the combined hematologic and neurologic abnormalities?''',
'Enterocyte metallothionein induction that preferentially retains copper, reducing systemic copper availability',[
('Inflammation-induced hepcidin elevation as the sole cause of restricted iron delivery','不能最好解释锌暴露、神经病变、中性粒细胞减少及特征性前体细胞空泡。'),
('Autoimmune loss of intrinsic factor with impaired cobalamin absorption','神经症状可相似，但本题B12和MMA正常且骨髓及暴露线索指向其他机制。'),
('Failure to replace iron losses from chronic gastrointestinal bleeding','铁状态、补铁无效和中性粒细胞/神经表现不支持孤立缺铁。')],
['medqa::0900','medqa::1009','medqa::0409','medqa-derived::A01','medqa-derived::A05'],
'复用补铁反应、铁代谢和B12代谢标志物对营养性贫血的排查；避免仅按神经表现复制旧病因。',
'锌相关铜吸收受阻及其骨髓和神经表型。',
'不能把所有贫血加感觉异常都归为B12缺乏，也不能仅凭环形铁粒幼细胞认定单一骨髓克隆疾病。',
'锌过量可通过肠细胞金属硫蛋白增加造成铜滞留和丢失；铜不足可同时造成贫血、中性粒细胞减少、前体细胞空泡及脊髓神经表现。',
['zinc_nih','copper_marrow'],['delayed_reuse','negative_transfer_control']),
]

ITEMS += [
item('R01','renal',
'''A 24-year-old man has persistent edema and 8.2 g of urinary protein per day. Serum albumin is 1.6 g/dL and creatinine was previously 0.9 mg/dL. A prior kidney biopsy showed diffuse podocyte foot-process effacement without immune deposits. He now develops abrupt left flank pain, visible hematuria, and a rise in creatinine to 1.7 mg/dL. Urine contains predominantly isomorphic erythrocytes without erythrocyte casts. Doppler and contrast imaging demonstrate a new left renal-vein thrombus. He has no previous thrombosis, no family history of thrombosis, and no malignancy. Which acquired change most directly links his pre-existing renal disease to the new complication?''',
'Urinary loss of antithrombin contributing to reduced endogenous anticoagulant activity',[
('New antibodies directed against glomerular basement membrane collagen','不能解释影像证实的肾静脉血栓与重度蛋白尿之间的主要因果联系。'),
('Loss of circulating fibrinogen leading to impaired fibrin formation','肾病综合征相关促凝状态不能由降低凝血能力来解释。'),
('Immune-complex activation of glomerular complement as the sole cause of the hematuria','血尿不自动表示新增免疫性肾小球炎；本题影像与尿沉渣提供了血管性定位。')],
['medqa::0574','medqa::0145','medqa::0697','medqa::1094'],
'复用大量蛋白尿、低白蛋白与足突病变的证据；用尿沉渣定位避免把血尿一概归入肾小球炎。',
'尿中抗凝蛋白丢失如何增加肾静脉血栓风险。',
'不能因新出现血尿就套用抗GBM或感染后肾炎的病因。',
'已证实的肾病综合征可造成获得性促凝状态，抗凝蛋白丢失是机制之一。血栓已由影像确诊，题目要求解释它与先前蛋白丢失的联系。',
['nephrotic_thrombosis'],['partial_reuse','condition_reversal']),

item('R02','renal',
'''A 17-year-old boy developed edema, cola-colored urine, erythrocyte casts, and low serum C3 two weeks after pharyngitis. An antistreptolysin O titer was elevated, and he initially improved with supportive care. Fourteen weeks later, microscopic hematuria and a urine protein/creatinine ratio of 1.8 g/g persist. C3 remains markedly low, whereas C4 is normal. Kidney biopsy shows dominant C3 staining with only trace immunoglobulin. Electron microscopy reveals ribbon-like, highly electron-dense intramembranous deposits. A circulating antibody is found that stabilizes an alternative-pathway convertase. Which effect best explains the persistent complement abnormality?''',
'Prolongation of C3bBb activity with continuing alternative-pathway C3 consumption',[
('Stabilization of C4b2a as the primary sustained event','C4b2a是经典/凝集素途径C3转化酶，不是本题C3肾炎因子相关的关键靶复合物。'),
('Direct antibody binding to the alpha-3 chain of type IV collagen','应重点考虑线状IgG的抗GBM过程，不能解释本题主导C3及致密膜内沉积。'),
('Selective loss of basement-membrane charge without complement activation','可关联蛋白滤过改变，但不能解释持续低C3及指定病理模式。')],
['medqa::0169','medqa::0533','medqa::0574'],
'组合感染后肾炎的时间线与C3肾炎因子、沉积物的证据，重新核对恢复过程。',
'持续低C3超出典型恢复窗口时，核实替代途径C3转化酶稳定化机制。',
'既往ASO阳性并不保证后续持续异常仍能由自限性感染后过程充分解释。',
'持续异常和C3主导病理要求重新评价补体失调。C3肾炎因子可稳定C3bBb，使替代途径持续活化；题目没有把单次低C3当成诊断充分条件。',
['c3_consensus'],['longitudinal_update','multi_memory_combination']),

item('R03','renal',
'''A 34-year-old woman with Graves disease has taken propylthiouracil for 20 months. She develops purpuric lesions, dyspnea, and hemoptysis. Creatinine has risen from 0.8 to 2.9 mg/dL, and urine contains dysmorphic erythrocytes and erythrocyte casts. Bronchoalveolar lavage becomes progressively bloodier. Anti-myeloperoxidase antibodies are strongly positive; anti-GBM antibodies and antinuclear antibodies are negative. C3 and C4 are normal. Kidney biopsy shows focal necrosis and cellular crescents, with little or no immunoglobulin or complement staining on immunofluorescence. Which mechanism best fits this combination of exposure, serology, and pathology?''',
'Drug-associated neutrophil-directed autoimmunity causing pauci-immune small-vessel injury',[
('Antibodies binding continuously along glomerular and alveolar basement membranes','肺肾表现可相似，但抗GBM阴性及寡免疫沉积不支持复制线状抗GBM机制。'),
('Persistent stabilization of the alternative-pathway C3 convertase','与补体正常、MPO阳性和寡免疫新月体病理的组合不符。'),
('Analgesic-mediated medullary ischemia with sloughing of renal papillae','不能解释肺泡出血、MPO抗体和肾小球新月体。')],
['medqa::0697','medqa::0162','medqa::0089','medqa::0512','medqa-derived::R02'],
'复用肺肾综合征的定位和免疫荧光对照，以及抗甲状腺药物暴露的记录。',
'PTU与MPO-ANCA相关血管炎及寡免疫肾损伤的联系。',
'不能把肺出血加红细胞管型直接等同于抗GBM病，也不能因旧题PTU适用就忽略新的药物损伤。',
'现有三类证据相互支持：长期PTU暴露、MPO阳性以及寡免疫坏死性新月体肾炎。正常补体与缺少线状沉积进一步限制旧病因的适用。',
['ptu_vasculitis','anti_gbm'],['multi_memory_combination','negative_transfer_control']),

item('R04','renal',
'''A 23-year-old man has had three episodes of visible hematuria during the past year. Each begins within 24 hours of sore throat or nasal congestion and resolves over several days. He has taken ibuprofen during each respiratory illness. Between episodes, urine still contains dysmorphic erythrocytes and occasional erythrocyte casts. Current urine protein/creatinine ratio is 0.9 g/g, serum albumin 4.0 g/dL, and creatinine 1.1 mg/dL. C3 and C4 have remained normal on repeated measurements. There is no hearing loss, purpura, or abdominal pain. Which kidney-biopsy finding would best account for the recurrent pattern?''',
'Dominant mesangial IgA deposition, often accompanied by C3',[
('Strong linear IgG staining along glomerular basement membranes','不能最好解释反复与上呼吸道症状同步的病程和目前缺少相应血清学/肺部证据的情况。'),
('Dominant intramembranous C3 deposits with minimal immunoglobulin','更需要持续补体失调等证据，不能沿用之前C3相关病例的结论。'),
('Coagulative necrosis of renal papillae without glomerular immune deposits','止痛药是干扰背景；畸形红细胞及红细胞管型更支持肾小球性出血。')],
['medqa::0169','medqa::0465','medqa::0216','medqa-derived::R02','medqa-derived::R03'],
'复用血尿定位、感染与发病间隔、补体和免疫荧光的鉴别框架。',
'同步性反复血尿与IgA优势系膜沉积的关联。',
'不能因有NSAID史就诊断乳头坏死，也不能因有咽痛就照搬延迟出现的感染后肾炎。',
'反复同步血尿、正常补体与肾小球性尿沉渣组合更符合IgA肾病；药物史单独不能推翻定位证据。',
['iga_review'],['condition_reversal','multi_memory_combination']),

item('R05','renal',
'''A 69-year-old man with progressive back pain and fatigue has hemoglobin 9.4 g/dL, creatinine 2.3 mg/dL, calcium 11.8 mg/dL, and serum albumin 3.5 g/dL. Skeletal imaging shows several lytic lesions. An adequately collected 24-hour urine sample contains 3.6 g of total protein, but only 0.18 g is albumin. The routine urine dipstick shows trace protein, while a sulfosalicylic acid precipitation test is strongly positive. Urine microscopy shows no dysmorphic erythrocytes or erythrocyte casts. Which explanation best reconciles the quantitative protein measurement with the weak dipstick result?''',
'Filtered monoclonal light chains exceed proximal reabsorptive capacity, while the dipstick preferentially detects albumin',[
('Selective glomerular albumin leakage due to isolated podocyte injury','定量已显示白蛋白只占很小部分，不符合选择性白蛋白丢失的解释。'),
('Transient orthostatic proteinuria with otherwise normal protein composition','持续肾功能异常及骨骼、血液和钙代谢表现不支持良性体位性过程。'),
('A dipstick false negative caused solely by severe hypoalbuminemia','本题血白蛋白未严重下降，关键是排出蛋白的种类和检测选择性。')],
['medqa::1153','medqa::0574','medqa::0145','medqa-derived::R01'],
'复用浆细胞相关溢出性蛋白尿与白蛋白为主的肾病综合征对照。',
'尿试纸检测偏好与总蛋白、白蛋白分量不一致时的解释。',
'不能仅因总蛋白达到肾病范围就套用足突/电荷屏障机制。',
'骨痛、高钙、贫血及非白蛋白为主的大量尿蛋白支持单克隆轻链溢出。弱试纸与强总蛋白检测并不矛盾，因为试纸主要检测白蛋白。',
['light_chain'],['delayed_reuse','multi_memory_combination']),

item('R06','renal',
'''A 67-year-old woman with a small lambda-restricted plasma-cell clone develops progressive edema and orthostatic dizziness. Creatinine is 1.8 mg/dL and serum albumin is 1.9 g/dL. Her 24-hour urine contains 6.4 g of protein, of which 5.7 g is albumin. Urine dipstick is 4+ positive, without erythrocyte casts. Kidney biopsy shows extracellular material expanding the mesangium and capillary walls. Electron microscopy reveals randomly arranged, nonbranching fibrils measuring approximately 9-11 nm. Which additional pathologic result would most strongly support the mechanism of her renal disease?''',
'Congo-red-positive glomerular deposits with apple-green birefringence and light-chain-derived fibril typing',[
('Predominantly distal tubular fractured light-chain casts without glomerular deposits','可见于轻链铸型损伤，但不能最好解释本例白蛋白为主的肾病综合征及肾小球纤维沉积。'),
('Diffuse podocyte effacement as the only abnormality, without extracellular deposits','与电镜所见纤维性细胞外沉积不符，不能仅按大量白蛋白尿复制微小病变。'),
('Granular IgG and C3 deposits containing large subepithelial immune-complex humps','沉积物形态和单克隆相关背景不支持该免疫复合物模式。')],
['medqa::1153','medqa::1094','medqa-derived::R01','medqa-derived::R05'],
'复用单克隆蛋白背景及蛋白尿成分定位，区分轻链溢出、肾小管损伤与肾小球病变。',
'纤维性肾小球沉积的染色和蛋白分型；不能仅凭存在单克隆克隆自动确定所有沉积类型。',
'相同浆细胞背景不意味着每次蛋白尿都来自轻链铸型或都以非白蛋白为主。',
'大量白蛋白尿定位于肾小球损伤，指定纤维形态支持淀粉样沉积。刚果红、偏振表现及沉积物分型共同支持轻链来源；仅有浆细胞克隆本身仍不足以完成分型。',
['renal_amyloid'],['condition_reversal','multi_memory_combination']),

item('R07','renal',
'''A 42-year-old woman began taking naproxen daily six weeks ago for chronic shoulder pain. She now has edema, a 7-kg weight gain, and reduced urine output. Creatinine has increased from 0.8 to 3.2 mg/dL, serum albumin is 1.8 g/dL, and 24-hour urine contains 7.9 g of protein, predominantly albumin. Urine has 3-5 leukocytes per high-power field but no erythrocyte casts. She has no fever, rash, peripheral eosinophilia, flank colic, or gross hematuria. Complement concentrations are normal. Ultrasonography excludes obstruction. Which combination of biopsy abnormalities would best explain the renal presentation?''',
'Interstitial inflammatory-cell infiltration together with diffuse podocyte foot-process effacement',[
('Necrotic sloughed papillae with otherwise intact glomerular filtration structures','可与镇痛药相关，但不能最好解释本题重度白蛋白尿与AKI的组合；缺少典型绞痛或肉眼血尿。'),
('Linear IgG deposition with widespread glomerular crescents','缺乏相应肾小球出血和血清学线索，不能因AKI就套用抗GBM病。'),
('Isolated mesangial IgA deposits with no tubulointerstitial or podocyte injury','不能最好解释药物时间线和显著肾病综合征。')],
['medqa::0145','medqa::1094','medqa::0465','medqa::0216','medqa-derived::R01','medqa-derived::R06'],
'复用白蛋白尿/足突损伤及镇痛药肾损伤的对照，先区分定位再判断病因。',
'NSAID相关间质性肾炎伴微小病变可缺少常见药物过敏表现。',
'不能把所有NSAID相关肾损伤都视为乳头坏死，也不能因无皮疹/嗜酸细胞增加就排除该类间质损伤。',
'NSAID相关间质性肾炎可伴足突病变和明显肾病综合征，且常缺乏典型全身过敏表现；两处损伤可共同解释本题，而单一乳头病变不足。',
['nsaid_nephrotic'],['multi_memory_combination','negative_transfer_control']),
]

ITEMS += [
item('T01','thyroid',
'''A term infant is born to a woman with Graves disease who required propylthiouracil during the third trimester. Maternal TSH-receptor antibodies were markedly elevated late in pregnancy. Cord free T4 is within the neonatal laboratory's reference interval, and the infant initially feeds normally. On day 8 he develops persistent tachycardia, sweating, frequent stools, and poor weight gain. Repeat testing shows suppressed TSH and free T4 above the age-specific reference interval; infant TSH-receptor antibodies are positive. He receives methimazole and symptomatic therapy. Which relationship best explains both the initially reassuring hormone measurement and the subsequent development of thyrotoxicosis?''',
'Transplacental antithyroid drug activity wanes before the stimulatory effect of maternally transferred IgG disappears',[
('Maternal IgM initially suppresses the fetal thyroid and then crosses the placenta after delivery','IgM不是这类经胎盘抗体效应的载体，且产后不存在新发生的胎盘转运。'),
('Maternal levothyroxine causes a delayed and permanently self-sustaining neonatal hormone excess','不符合所给药物暴露、抗体检测及暂时性免疫机制。'),
('A normal cord free T4 rules out antibody-related disease, so the later findings require a new neonatal autoimmune disorder','母体药物可影响初始激素表现，单次正常结果不能排除后续风险。')],
['medqa::0067','medqa::0361','medqa::0089'],
'组合母体抗体与新生儿处理框架，同时区分母体用药效应和抗体效应。',
'不同作用持续时间如何解释正常初始检查后的延迟发病。',
'不能因母体接受治疗或出生时激素正常就认为新生儿不再受刺激性抗体影响。',
'母体抗甲状腺药与刺激性IgG可同时影响新生儿。药物效应消退后，仍存在的刺激性抗体可使甲亢显露；出生时正常的单次激素结果并不充分排除。',
['neonatal_endo'],['longitudinal_update','multi_memory_combination']),

item('T02','thyroid',
'''An 8-month-old boy has required continuous antithyroid treatment since the first weeks of life. During two carefully monitored dose reductions, free T4 rises again despite persistently suppressed TSH. The thyroid is diffusely enlarged. TSH-receptor antibodies were undetectable in cord blood and remain undetectable in both infant and mother on repeated testing. His mother has no thyroid disease. His father and paternal aunt both developed hyperthyroidism in early childhood without orbitopathy or detectable TSH-receptor antibodies. The child has no cafe-au-lait lesions or skeletal abnormalities. Which mechanism best accounts for this course?''',
'A germline activating variant of the TSH receptor producing constitutive thyroid stimulation',[
('Prolonged stimulation by maternally transferred receptor antibodies','持续时间、反复抗体阴性及父系家族模式均不支持。'),
('A germline inactivating variant of the TSH receptor causing reduced receptor signaling','主要与受体信号不足相关，不能解释反复激素升高。'),
('Transient escape from an iodine-induced block of neonatal thyroid synthesis','不符合长期复发及多个父系亲属早发疾病的模式。')],
['medqa::0067','medqa::0124','medqa-derived::T01'],
'复用经胎盘刺激性抗体效应的时间边界及先天受体功能异常与功能状态的关系。',
'持续、抗体阴性且父系聚集的甲亢所提示的构成性受体激活。',
'不能把所有新生儿起病甲亢都归于母体Graves；此处无母体抗体的证据且病程超出暂时性模式。',
'父系家族聚集、持续复发和反复抗体阴性支持遗传性非自身免疫性甲亢。激活性TSHR变异可产生与TSH无关的信号。',
['neonatal_endo'],['condition_reversal','negative_transfer_control']),

item('T03','thyroid',
'''A 29-year-old woman at 11 weeks of gestation has Graves disease. She was switched from methimazole to propylthiouracil five weeks ago. She now presents with fever of 39.2 degrees C, a painful sore throat, and several oral ulcers. Heart rate is 106/min, blood pressure 114/72 mm Hg, and she is alert. Free T4 is only mildly above the pregnancy-specific reference interval, and liver tests are normal. Leukocyte count is 900/mm3 with an absolute neutrophil count of 70/mm3. She has not received chemotherapy or another marrow-toxic medication. Which immediate strategy is most appropriate?''',
'Stop propylthiouracil and begin urgent evaluation and empiric antimicrobial management for febrile agranulocytosis',[
('Immediately substitute methimazole because ongoing pregnancy makes uninterrupted thionamide therapy mandatory','严重粒细胞缺乏时不应机械换用另一硫脲类药物，既往孕期换药规则不能覆盖本次重大不良反应。'),
('Continue propylthiouracil while repeating the blood count in one week','发热伴极低ANC需要立即处理，不能继续疑似致病药并等待。'),
('Administer radioactive iodine now to eliminate the need for medication during pregnancy','妊娠期不适用，且无法取代对当前发热性粒细胞缺乏的紧急处理。')],
['medqa::0089','medqa::0512','medqa::1061','medqa-derived::R03'],
'复用孕期治疗边界及PTU重大不良反应需重审适用性的经验；区分发热的原因。',
'发热性粒细胞缺乏的紧急药物安全处理及同类药替换限制。',
'不能因早孕时PTU曾是合适选择就继续使用，也不能把所有发热和心动过速当作甲状腺危象。',
'ANC 70伴发热和咽部症状构成紧急风险，需停止疑似致病药并按发热性粒细胞缺乏处理。孕期常规换药策略不能凌驾于重大药物不良反应。',
['atd_safety'],['condition_reversal','multi_memory_combination']),

item('T04','thyroid',
'''A 32-year-old woman develops palpitations and heat intolerance eight weeks after delivery. She is breastfeeding and has never used thyroid medication. TSH is suppressed and free T4 is elevated. Thyroid-peroxidase antibodies are positive, TSH-receptor antibodies are negative, and Doppler ultrasonography shows low thyroid blood flow. There is no thyroid bruit or orbitopathy. Without antithyroid medication, the palpitations resolve. Ten weeks later she develops constipation and fatigue, with free T4 below the reference interval and TSH 16 mIU/L. Which explanation best accounts for why methimazole would not have targeted the primary abnormality during the initial phase?''',
'The initial hormone excess resulted predominantly from release of stored hormone from injured follicles rather than increased synthesis',[
('The initial excess was sustained by receptor-antibody-driven synthesis that is intrinsically resistant to methimazole','抗体、血流和双相病程不支持以Graves合成为主的解释。'),
('Persistently elevated placental hCG remained the dominant stimulus throughout both postpartum phases','分娩后的时间线和随后功能减低不能由持续胎盘刺激统一解释。'),
('The initial excess was produced solely by accelerated peripheral T4-to-T3 conversion with no thyroid injury','不能解释局部血流、抗体背景及随后的低功能阶段。')],
['medqa::0089','medqa::0137','medqa::0229','medqa-derived::T01'],
'复用甲状腺毒症不等于某一固定病因的框架，重新评估抗体及合成抑制药的靶点。',
'产后破坏性过程的低血流、抗体及双相病程证据。',
'不能因存在甲状腺毒症就自动复用Graves的合成抑制方案。',
'低血流、TRAb阴性及自行由毒症转为低功能共同支持破坏性激素释放；抑制新合成不能直接阻断已经储存激素的释放。',
['postpartum_primary','postpartum_endo'],['longitudinal_update','condition_reversal']),

item('T05','thyroid',
'''A 54-year-old man with previous surgery and radiotherapy for a pituitary macroadenoma is brought in with hypothermia and confusion after a respiratory infection. He stopped attending endocrine follow-up two years ago and takes no hormone replacement. Temperature is 34.0 degrees C, pulse 46/min, and blood pressure 78/46 mm Hg. Sodium is 121 mEq/L, glucose 48 mg/dL, free T4 0.22 ng/dL (reference 0.8-1.8), and TSH 1.4 mIU/L (reference 0.4-4.5). An 08:00 cortisol sample obtained before treatment is 1.6 micrograms/dL, with a low ACTH concentration. Airway support, cautious fluids, dextrose, and infection evaluation have begun. Which hormone-treatment sequence most appropriately addresses the immediate risk?''',
'Give intravenous hydrocortisone before or with initiation of intravenous levothyroxine',[
('Give levothyroxine alone and reassess the cortisol deficiency after thyroid indices normalize','启动甲状腺激素可能加重未覆盖的肾上腺功能不足，不能推迟糖皮质激素。'),
('Withhold both hormones because a reference-range TSH establishes nonthyroidal illness','垂体病史下，正常范围TSH并不代表对极低FT4的适当反应。'),
('Give fludrocortisone alone because the low sodium proves isolated mineralocorticoid deficiency','不能纠正明确的中枢性皮质醇和甲状腺轴不足，也不能仅由低钠推断孤立盐皮质激素缺乏。')],
['medqa::0103','medqa::0229','medqa-derived::T04'],
'复用严重低功能与非甲状腺疾病综合征的对照，按病程及FT4/TSH适当性而非TSH是否超参考区间判断。',
'垂体多轴不足时甲状腺激素与糖皮质激素的安全给药顺序。',
'不能把旧重症病例的正常TSH和低T3结论泛化到有明确垂体损伤及严重低FT4/低皮质醇的新病例。',
'极低FT4伴不适当正常TSH及垂体病史支持中枢性功能不足。严重低皮质醇与休克要求先给予应激糖皮质激素覆盖，再启动或同时启动甲状腺激素。',
['myxedema_ata','myxedema_endo'],['multi_memory_combination','condition_reversal']),

item('T06','thyroid',
'''A 38-year-old woman has persistent palpitations, weight loss, and intermittent right pelvic discomfort. TSH is below 0.01 mIU/L and free T4 remains elevated on repeated testing over four months. TSH-receptor antibodies are negative. She is not pregnant or postpartum and has had no recent iodinated contrast, amiodarone, or prescribed thyroid hormone. She does use an over-the-counter product marketed for thyroid support. Serum thyroglobulin is elevated, with negative thyroglobulin antibodies. Neck ultrasonography shows a small, otherwise unremarkable thyroid, and 24-hour cervical radioiodine uptake is 0.4%. Pelvic ultrasonography reveals a 6-cm complex right adnexal mass. Which additional finding would most specifically establish an endogenous source of her persistent hormone excess?''',
'Focal radionuclide uptake within the adnexal mass on whole-body scintigraphy',[
('A positive thyroid-peroxidase antibody result by itself','不能定位激素来源，也不能单凭这一抗体区分所有低摄取甲状腺毒症。'),
('Low thyroid blood flow on Doppler imaging of the neck','已与颈部低摄取一致，但不能证明何处持续合成激素。'),
('Suppression of pituitary TSH during symptomatic episodes','这是已有甲状腺毒症的轴反应，不能区分外源、破坏性和异位来源。')],
['medqa::0396','medqa::0137','medqa-derived::T04','medqa-derived::T05'],
'组合颈部检查与功能状态分离、低摄取病因及异位甲状腺组织的索引。',
'全身核素定位与甲状腺球蛋白的来源鉴别价值；仅颈部摄取低不能完成定位。',
'不能把低颈部摄取自动等同于甲状腺炎，也不能因补充剂暴露就忽略内源性异位来源。',
'颈部低摄取可有多种病因；在本例临床背景下，附件肿块内功能性核素摄取最直接定位内源性激素来源。球蛋白升高是支持线索，不单独作为定位依据。',
['struma_endo','struma_scan'],['delayed_reuse','multi_memory_combination']),
]
