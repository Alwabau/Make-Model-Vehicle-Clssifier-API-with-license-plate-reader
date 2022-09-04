'''
Utility funcion to merge subclasses into the main class
'''
def class_mapping(string):
    '''
    this function maps a subclass string to main class string
    '''
    classes_dict = {    'abarth_500c': 'abarth_500',
                        'abarth_595': 'abarth_500',
                        'abarth_595-competizione': 'abarth_500',
                        'abarth_595-turismo': 'abarth_500',
                        'abarth_595c': 'abarth_500',
                        'bmw_114': 'bmw_100', 'bmw_116': 'bmw_100', 'bmw_118': 'bmw_100', 'bmw_120': 'bmw_100', 'bmw_123': 'bmw_100', 'bmw_125': 'bmw_100',
                        'bmw_130': 'bmw_100','bmw_135': 'bmw_100','bmw_140': 'bmw_100',
                        'bmw_214': 'bmw_200','bmw_216': 'bmw_200','bmw_218': 'bmw_200','bmw_220': 'bmw_200','bmw_225': 'bmw_200',
                        'bmw_228': 'bmw_200','bmw_230': 'bmw_200','bmw_235': 'bmw_200','bmw_240': 'bmw_200',
                        'bmw_316': 'bmw_300','bmw_318': 'bmw_300','bmw_320': 'bmw_300','bmw_323': 'bmw_300','bmw_325': 'bmw_300',
                        'bmw_328': 'bmw_300','bmw_330': 'bmw_300','bmw_335': 'bmw_300','bmw_340': 'bmw_300',
                        'bmw_418': 'bmw_400','bmw_420': 'bmw_400','bmw_425': 'bmw_400','bmw_428': 'bmw_400',
                        'bmw_430': 'bmw_400','bmw_435': 'bmw_400','bmw_440': 'bmw_400',
                        'bmw_518': 'bmw_500','bmw_520': 'bmw_500','bmw_523': 'bmw_500','bmw_525' : 'bmw_500','bmw_528': 'bmw_500','bmw_530': 'bmw_500','bmw_535': 'bmw_500',
                        'bmw_540': 'bmw_500','bmw_545': 'bmw_500','bmw_550': 'bmw_500',
                        'bmw_630': 'bmw_600','bmw_635': 'bmw_600','bmw_645': 'bmw_600','bmw_640': 'bmw_600',
                        'bmw_450': 'bmw_600','bmw_650': 'bmw_600',
                        'bmw_730': 'bmw_700','bmw_735': 'bmw_700','bmw_740': 'bmw_700','bmw_750': 'bmw_700','bmw_745': 'bmw_700',
                        'citroen_c3-aircross': 'citroen_c3','citroen_c3-picasso': 'citroen_c3',
                        'citroen_c4-aircross': 'citroen_c4','citroen_c4-cactus': 'citroen_c4','citroen_c4-picasso': 'citroen_c4','citroen_c4-spacetourer': 'citroen_c4',
                        'citroen_grand-c4-picasso' : 'citroen_c4', 'citroen_grand-c4-spacetourer' : 'citroen_c4',
                        'citroen_grand-c4' : 'citroen_c4',
                        'fiat_500c': 'fiat_500', 'fiat_500l': 'fiat_500', 'fiat_500x': 'fiat_500',
                        'ford_focus-c-max':'ford_focus', 'ford_focus-c-max-coupe':'ford_focus', 'ford_focus-cc':'ford_focus',
                        'ford_tourneo-connect': 'ford_tourneo', 'ford_tourneo-courier': 'ford_tourneo', 'ford_tourneo-custom' : 'ford_tourneo',
                        'ford_transit-connect': 'ford_transit', 'ford_transit-custom': 'ford_transit', 'ford_transit-courier': 'ford_transit', 
                        'land-rover_range-rover-evoque': 'land_rover_range-rover', 'land-rover_range-rover-sport': 'land_rover_range-rover',
                        'land-rover_range-rover-velar': 'land_rover_range-rover',
                        'land-rover_discovery-sport': 'land-rover_discovery',
                        'mercedes-benz_a-220':'mercedes-benz_a-200', 'mercedes-benz_a-250':'mercedes-benz_a-200',
                        'mercedes-benz_a-140':'mercedes-benz_a-100','mercedes-benz_a-150':'mercedes-benz_a-100','mercedes-benz_a-160':'mercedes-benz_a-100',
                        'mercedes-benz_a-170':'mercedes-benz_a-100','mercedes-benz_a-190':'mercedes-benz_a-100','mercedes-benz_a-180':'mercedes-benz_a-100',
                        'mercedes-benz_b-150':'mercedes-benz_b-100','mercedes-benz_b-160':'mercedes-benz_b-100','mercedes-benz_b-170':'mercedes-benz_b-100',
                        'mercedes-benz_b-180':'mercedes-benz_b-100',
                        'mercedes-benz_b-220':'mercedes-benz_b-200','mercedes-benz_b-250':'mercedes-benz_b-200',
                        'mercedes-benz_c-220':'mercedes-benz_c-200','mercedes-benz_c-230':'mercedes-benz_c-200',
                        'mercedes-benz_c-240':'mercedes-benz_c-200','mercedes-benz_c-250':'mercedes-benz_c-200',
                        'mercedes-benz_c-270':'mercedes-benz_c-200','mercedes-benz_c-280':'mercedes-benz_c-200',
                        'mercedes-benz_c-320':'mercedes-benz_c-300','mercedes-benz_c-350':'mercedes-benz_c-300',
                        'mercedes-benz_c-43-amg': 'mercedes-benz_c-400', 'mercedes-benz_c-450': 'mercedes-benz_c-400',
                        'mercedes-benz_cla-220':'mercedes-benz_cla-200', 'mercedes-benz_cla-250':'mercedes-benz_cla-200',
                        'mercedes-benz_clk-200':'mercedes-benz_clk','mercedes-benz_clk-220':'mercedes-benz_clk',
                        'mercedes-benz_clk-240':'mercedes-benz_clk','mercedes-benz_clk-270':'mercedes-benz_clk',
                        'mercedes-benz_clk-280':'mercedes-benz_clk','mercedes-benz_clk-230':'mercedes-benz_clk',
                        'mercedes-benz_clk-320':'mercedes-benz_clk','mercedes-benz_clk-350':'mercedes-benz_clk',
                        'mercedes-benz_cls-220':'mercedes-benz_cls','mercedes-benz_cls-250':'mercedes-benz_cls',
                        'mercedes-benz_cls-320':'mercedes-benz_cls','mercedes-benz_cls-350':'mercedes-benz_cls',
                        'mercedes-benz_cls-400':'mercedes-benz_cls','mercedes-benz_cls-450':'mercedes-benz_cls',
                        'mercedes-benz_cls-500':'mercedes-benz_cls',
                        'mercedes-benz_e-200':'mercedes-benz_e','mercedes-benz_e-220':'mercedes-benz_e',
                        'mercedes-benz_e-240':'mercedes-benz_e','mercedes-benz_e-250':'mercedes-benz_e',
                        'mercedes-benz_e-270':'mercedes-benz_e','mercedes-benz_e-280':'mercedes-benz_e',
                        'mercedes-benz_e-300':'mercedes-benz_e','mercedes-benz_e-320':'mercedes-benz_e',
                        'mercedes-benz_e-350':'mercedes-benz_e','mercedes-benz_e-400':'mercedes-benz_e',
                        'mercedes-benz_e-43-amg':'mercedes-benz_e','mercedes-benz_e-500':'mercedes-benz_e',
                        'mercedes-benz_e-55-amg':'mercedes-benz_e','mercedes-benz_e-63-amg':'mercedes-benz_e',
                        'mercedes-benz_g-320':'mercedes-benz_g','mercedes-benz_g-350':'mercedes-benz_g',
                        'mercedes-benz_g-400':'mercedes-benz_g','mercedes-benz_g-500':'mercedes-benz_g',
                        'mercedes-benz_g-63-amg':'mercedes-benz_g',
                        'mercedes-benz_gl-320':'mercedes-benz_gl','mercedes-benz_gl-350':'mercedes-benz_gl',
                        'mercedes-benz_gla-180':'mercedes-benz_gla','mercedes-benz_gla-200':'mercedes-benz_gla',
                        'mercedes-benz_gla-220':'mercedes-benz_gla','mercedes-benz_gla-250':'mercedes-benz_gla',
                        'mercedes-benz_gla-45-amg':'mercedes-benz_gla',
                        'mercedes-benz_glc-220':'mercedes-benz_glc','mercedes-benz_glc-250':'mercedes-benz_glc',
                        'mercedes-benz_glc-43-amg':'mercedes-benz_glc','mercedes-benz_glc-350':'mercedes-benz_glc',
                        'mercedes-benz_glc-63-amg':'mercedes-benz_glc','mercedes-benz_glc-250':'mercedes-benz_glc',
                        'mercedes-benz_glc-300':'mercedes-benz_glc',
                        'mercedes-benz_gle-250':'mercedes-benz_gle','mercedes-benz_gle-350':'mercedes-benz_gle',
                        'mercedes-benz_gle-400':'mercedes-benz_gle','mercedes-benz_gle-43-amg':'mercedes-benz_gle',
                        'mercedes-benz_gle-450':'mercedes-benz_gle','mercedes-benz_gle-500':'mercedes-benz_gle',
                        'mercedes-benz_gle-63-amg':'mercedes-benz_gle',
                        'mercedes-benz_glk-200':'mercedes-benz_glk','mercedes-benz_glk-220':'mercedes-benz_glk',
                        'mercedes-benz_glk-250':'mercedes-benz_glk','mercedes-benz_glk-320':'mercedes-benz_glk',
                        'mercedes-benz_glk-350':'mercedes-benz_glk',
                        'mercedes-benz_ml-250':'mercedes-benz_ml','mercedes-benz_ml-270':'mercedes-benz_ml',
                        'mercedes-benz_ml-280':'mercedes-benz_ml','mercedes-benz_ml-300':'mercedes-benz_ml',
                        'mercedes-benz_ml-320':'mercedes-benz_ml','mercedes-benz_ml-350':'mercedes-benz_ml',
                        'mercedes-benz_ml-400':'mercedes-benz_ml','mercedes-benz_ml-420':'mercedes-benz_ml',
                        'mercedes-benz_ml-63-amg':'mercedes-benz_ml',
                        'mercedes-benz_r-280':'mercedes-benz_r','mercedes-benz_r-320':'mercedes-benz_r',
                        'mercedes-benz_r-350':'mercedes-benz_r',
                        'mercedes-benz_s-320':'mercedes-benz_s','mercedes-benz_s-350':'mercedes-benz_s',
                        'mercedes-benz_s-400':'mercedes-benz_s','mercedes-benz_s-420':'mercedes-benz_s',
                        'mercedes-benz_s-500':'mercedes-benz_s','mercedes-benz_s-560':'mercedes-benz_s',
                        'mercedes-benz_s-63-amg':'mercedes-benz_s',
                        'mercedes-benz_sl-320':'mercedes-benz_sl','mercedes-benz_sl-350':'mercedes-benz_sl',
                        'mercedes-benz_sl-500':'mercedes-benz_sl','mercedes-benz_sl-55-amg':'mercedes-benz_sl',
                        'mercedes-benz_slc-180':'mercedes-benz_slc','mercedes-benz_slc-200':'mercedes-benz_slc',
                        'mercedes-benz_slc-250':'mercedes-benz_slc','mercedes-benz_slc-43-amg':'mercedes-benz_slc',
                        'mercedes-benz_slk-200':'mercedes-benz_slk','mercedes-benz_slk-230':'mercedes-benz_slk',
                        'mercedes-benz_slk-280':'mercedes-benz_slk','mercedes-benz_slk-350':'mercedes-benz_slk',
                        'mercedes-benz_slk-250':'mercedes-benz_slk',
                        'mercedes-benz_v-220':'mercedes-benz_v','mercedes-benz_v-250':'mercedes-benz_v',
                        'mercedes-benz_vaneo':'mercedes-benz_v','mercedes-benz_viano':'mercedes-benz_v',
                        'mercedes-benz_vito':'mercedes-benz_v',
                        'mini_cooper-d':'mini_cooper','mini_cooper-s':'mini_cooper',
                        'mini_cooper-sd':'mini_cooper','mini_john-cooper-works':'mini_cooper',
                        'mini_one':'mini_cooper','mini_john-cooper-works':'mini_cooper',
                        'mini_one-d':'mini_cooper',
                        'mini_cooper-d-cabrio':'mini_cooper-cabrio','mini_cooper-s-cabrio':'mini_cooper-cabrio',
                        'mini_john-cooper-works-cabrio':'mini_cooper-cabrio','mini_one-cabrio':'mini_cooper-cabrio',
                        'mini_cooper-d-clubman':'mini_cooper-clubman','mini_cooper-s-clubman':'mini_cooper-clubman',
                        'mini_cooper-sd-clubman':'mini_cooper-clubman','mini_john-cooper-works-clubman':'mini_cooper-clubman',
                        'mini_one-d-clubman':'mini_cooper-clubman',
                        'mini_cooper-d-countryman':'mini_cooper-countryman','mini_cooper-s-countryman':'mini_cooper-countryman',
                        'ini_cooper-sd-countryman':'mini_cooper-countryman','mini_john-cooper-works-countryman':'mini_cooper-countryman',
                        'mini_one-countryman':'mini_cooper-countryman','mini_cooper-sd-countryman':'mini_cooper-countryman',
                        'mini_one-d-countryman':'mini_cooper-countryman',
                        'mini_cooper-d-paceman':'mini_cooper-paceman','mini_cooper-s-paceman':'mini_cooper-paceman',
                        'mini_cooper-sd-paceman':'mini_cooper-paceman',
                        'mini_cooper-s-roadster':'mini_cooper-roadster',
                        'volkswagen_golf-cabriolet':'volkswagen_golf','volkswagen_golf-gti':'volkswagen_golf',
                        'volkswagen_golf-plus':'volkswagen_golf','volkswagen_golf-sportsvan':'volkswagen_golf',
                        'volkswagen_golf-variant':'volkswagen_golf',
                        'volkswagen_passat-alltrack':'volkswagen_passat','volkswagen_passat-cc':'volkswagen_passat',
                        'volkswagen_passat-variant':'volkswagen_passat',
                        'volkswagen_polo-cross':'volkswagen_polo','volkswagen_polo-gti':'volkswagen_polo',
                        'volkswagen_t4-caravelle':'volkswagen_t4','volkswagen_t4-multivan':'volkswagen_t4',
                        'volkswagen_t5-california':'volkswagen_t5','volkswagen_t5-caravelle':'volkswagen_t5',
                        'volkswagen_t5-kombi':'volkswagen_t5','volkswagen_t5-multivan':'volkswagen_t5',
                        'volkswagen_t5-shuttle':'volkswagen_t5',
                        'volkswagen_t6-california':'volkswagen_t6','volkswagen_t6-caravelle':'volkswagen_t6',
                        'volkswagen_t6-multivan':'volkswagen_t6','volkswagen_t6-transporter':'volkswagen_t6',
        }
    if string in classes_dict:
        return classes_dict[string]
    else:
        return string

#This function will look for a class in a given list, will return -1 if class not found
def get_model_index(models_list, model):
  try:
    return models_list.index(model) 
  except:
    return -1

def get_merged_lists(models, predicts):
    out_models = []
    out_preds = []
    for i in range(len(models)):
        make_model = class_mapping(models[i])
        class_index = get_model_index(out_models,make_model)
        if class_index > -1:
            out_preds[class_index] += predicts[i]
        else:
            out_models.append(make_model)
            out_preds.append(predicts[i])
    return out_models, out_preds
