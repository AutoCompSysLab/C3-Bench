"""Change-detection criteria and prompt templates for C3-Bench.

For every domain/topic this module defines the *change criteria* that tell the
model which visual changes to report and which variations to ignore, together
with the inference templates (criteria-guided and naive) and the LLM-judge
template used for evaluation.

The ``*_change_criteria`` string constants form a catalog that is collected into
:data:`CHANGE_CRITERIA` for programmatic lookup via
:func:`resolve_change_criteria`. The template strings are kept verbatim so that
generated prompts remain byte-for-byte reproducible.
"""

import re

natural_scenes_cantina_change_criteria = '''
## Change to Detect
    -   Movement, appearance, disappearance, or modification of objects, tools, or fixtures
    -   Structural or spatial rearrangements within the scene
    -   Functional or environmental state changes (e.g., doors, drawers, or appliances opened/closed)
    
## No Change
    -   Lighting or exposure variations
    -   Viewpoint or camera shifts
    -   Minor photometric or compression artifacts
    
## Tone
    -   Neutral and objective
'''

natural_scenes_construction_site_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of humans, vehicles, fences, construction equipment, or road signs
    - Structural or spatial rearrangements of materials, barriers, or machinery within the site
    - Functional or environmental state changes (e.g., barriers opened/closed, machinery activated/deactivated)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Seasonal, weather, or shadow differences

## Tone
    - Neutral and objective
'''

natural_scenes_crosswalk_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of pedestrians, vehicles, traffic lights, or crosswalk signs
    - Structural or spatial rearrangements of barriers, signals, or road markings within the scene
    - Functional or environmental state changes (e.g., traffic lights turning on/off, pedestrian lights changing states)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Weather, seasonal, or shadow differences

## Tone
    - Neutral and objective
'''

natural_scenes_highway_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of vehicles, traffic signs, cones, or roadside objects
    - Structural or spatial rearrangements of lanes, barriers, or road markings
    - Functional or environmental state changes (e.g., lane closures, traffic signs activated/deactivated)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Weather, seasonal, or shadow differences

## Tone
    - Neutral and objective
'''

natural_scenes_lounge_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of furniture, chairs, tables, cups, or other indoor objects
    - Structural or spatial rearrangements of interior elements (e.g., furniture layout, partitions, or decorations)
    - Functional or environmental state changes (e.g., lights turned on/off, screens or displays showing different content)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Weather, shadow, or reflection differences

## Tone
    - Neutral and objective
'''

natural_scenes_lunch_room_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of dishes, food items, cups, trays, or utensils
    - Rearrangement or relocation of tables, chairs, or other furniture
    - Presence or absence of people or personal belongings (e.g., bags, bottles, jackets)
    - Functional or environmental state changes (e.g., appliances turned on/off, screens or lights activated)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Shadows, reflections, or time-of-day differences

## Tone
    - Neutral and objective
'''

natural_scenes_meeting_room_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of chairs, tables, laptops, documents, or presentation equipment
    - Presence or absence of people or personal items (e.g., notebooks, pens, bottles)
    - Structural or spatial rearrangements within the room (e.g., furniture layout, screen position)
    - Functional or environmental state changes (e.g., monitor or projector turned on/off, lights or blinds adjusted)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Shadows, reflections, or time-of-day differences

## Tone
    - Neutral and objective
'''

natural_scenes_parking_lot_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of vehicles, pedestrians, cones, fences, or parking signs
    - Structural or spatial rearrangements of parking lines, barriers, or lot elements
    - Functional or environmental state changes (e.g., gates opened/closed, lights or signals turned on/off)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Weather, seasonal, or shadow differences

## Tone
    - Neutral and objective
'''

natural_scenes_printing_area_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of printers, paper stacks, office supplies, or printed documents
    - Presence or absence of people or personal items (e.g., notebooks, cups, bags)
    - Structural or spatial rearrangements of desks, shelves, or equipment
    - Functional or environmental state changes (e.g., printer or copier turned on/off, trays opened/closed, paper inserted/removed)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Shadows, reflections, or time-of-day differences

## Tone
    - Neutral and objective
'''

natural_scenes_railway_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of trains, vehicles, pedestrians, or maintenance equipment
    - Structural or spatial changes to tracks, signals, fences, or station-related facilities
    - Functional or operational state changes (e.g., signal lights on/off, barriers opened/closed, electronic boards updated)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Weather, seasonal, or shadow differences

## Tone
    - Neutral and objective
'''

natural_scenes_residential_yard_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of vehicles, people, furniture, plants, or gardening tools
    - Structural or spatial rearrangements of fences, decorations, or outdoor equipment
    - Functional or environmental state changes (e.g., doors, gates, or windows opened/closed, lights turned on/off)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Weather, seasonal, or shadow differences

## Tone
    - Neutral and objective
'''

natural_scenes_sidewalk_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of pedestrians, vehicles, street signs, or sidewalk objects (e.g., benches, trash cans, kiosks)
    - Structural or spatial rearrangements of barriers, poles, or pavement features
    - Functional or environmental state changes (e.g., lights or digital displays turned on/off, shop doors opened/closed)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Weather, seasonal, or shadow differences

## Tone
    - Neutral and objective
'''

natural_scenes_street_road_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of vehicles, pedestrians, traffic lights, street signs, or roadside objects
    - Structural or spatial rearrangements of lanes, barriers, or road markings
    - Functional or environmental state changes (e.g., signals activated/deactivated, shop doors or shutters opened/closed)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Weather, seasonal, or shadow differences

## Tone
    - Neutral and objective
'''

natural_scenes_warehouse_change_criteria = '''
## Change to Detect
    - Movement, appearance, disappearance, or modification of boxes, fences, containers, or storage equipment
    - Rearrangement or relocation of pallets, machinery, or shelving structures
    - Functional or environmental state changes (e.g., doors or gates opened/closed, conveyor belts activated/deactivated)

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Weather, shadow, or reflection differences

## Tone
    - Neutral and objective
'''

anomalies_bottle_change_criteria = '''
## Change to Detect
    - Appearance, disappearance, or modification of cracks, breaks, or missing parts on the bottle surface
    - Presence of contamination, stains, or foreign substances
    - Deformation, dent, or structural damage of the bottle body or cap
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or reflection differences
    - Color tone or shading variations not related to physical damage

## Tone
    - Neutral and objective
'''

anomalies_cable_change_criteria = '''
## Change to Detect
    - Bending or deformation of wires or outer insulation
    - Swapped or misplaced cables
    - Cut or broken sections of inner or outer insulation
    - Missing cables or disconnected wires
    - Poked insulation or exposed internal wires
    - Combined or multiple occurrences of the above defect types
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or reflection differences
    - Color tone or shading variations not related to physical damage

## Tone
    - Neutral and objective
'''

anomalies_candle_change_criteria = '''
## Change to Detect
    - Presence of contamination, stains, or foreign substances on the candle surface
    - Surface damage such as cracks, scratches, or dents on the wax or label area
    - Deformation or melting-related irregularities caused by physical alteration
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or reflection differences
    - Color tone or shading variations not related to actual surface damage

## Tone
    - Neutral and objective
'''

anomalies_capsule_change_criteria = '''
## Change to Detect
    - Crack: visible fractures or splits on the capsule surface
    - Fault imprint: unintended embossed/debossed marks or misprints on text/logo
    - Poke: punctures or holes penetrating the shell
    - Scratch: linear abrasions or scuffs on coating or label area
    - Squeeze: deformation/warping due to compression (flattened or distorted shape)
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or reflection differences
    - Color tone or shading variations not caused by physical damage

## Tone
    - Neutral and objective
'''

anomalies_capsule_multiple_change_criteria = '''
## Change to Detect
    - Deformation: warping, dents, flattening, or misshapen shell
    - Leakage: liquid/powder residue, seepage, stains, or wet marks from contents
    - Surface damage: cracks, scratches, scuffs, punctures, or chipped coating
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or compression artifacts
    - Color tone or reflections not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_carpet_change_criteria = '''
## Change to Detect
    - Color anomalies such as discoloration, fading, or irregular color patches
    - Cuts or tears in the fabric surface
    - Holes or missing fiber areas
    - Metal contamination or presence of foreign metallic objects
    - Thread damage, fraying, or structural distortion in the weave pattern
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or reflection differences
    - Color tone or shading variations not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_cashew_change_criteria = '''
## Change to Detect
    - Dent or deformation of the nut surface or shell
    - Discoloration or uneven color patches on the surface
    - Mold, stains, or foreign contamination
    - Overlapping or clumped nuts forming irregular clusters
    - Surface damage such as scratches, cracks, or chipping
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or reflection differences
    - Color tone variations not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_cigarette_box_change_criteria = '''
## Change to Detect
    - Opened or partially opened cigarette boxes
    - Visible displacement or lifting of the box lid or flap
    - Any structural state change indicating the box is no longer sealed or closed
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric or reflection differences
    - Color tone variations not caused by physical state change

## Tone
    - Neutral and objective
'''

anomalies_drink_bottle_change_criteria = '''
## Change to Detect
    - Cap partially opened (half-open) or fully opened
    - Structural deformation or detachment of the bottle cap
    - Surface damage such as scratches, dents, cracks, or abrasions on the bottle body or label area
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression differences
    - Color tone or shading variations not caused by physical damage

## Tone
    - Neutral and objective
'''

anomalies_drink_can_change_criteria = '''
## Change to Detect
    - Deformation: dents, warping, crushed or misshapen can body or rim
    - Straw missing/presence: absence of an expected straw or dislodged/misplaced straw
    - Surface damage: scratches, scuffs, cracks, punctures, or coating/print chipping
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_food_bottle_change_criteria = '''
## Change to Detect
    - Deformation: dents, warping, misshapen bottle or cap
    - Opened state: cap loosened/half-open or fully opened; seal broken
    - Surface damage: scratches, scuffs, cracks, punctures, or label/coating chipping
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_food_box_change_criteria = '''
## Change to Detect
    - Deformation: dents, warping, or misshapen box structure
    - Opened state: box lid partially or fully opened; seal or flap broken
    - Surface damage: scratches, scuffs, tears, punctures, or label/coating peeling
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_food_package_change_criteria = '''
## Change to Detect
    - Broken: tears, rips, punctures, split seams, or broken seals/flaps
    - Surface anomaly: stains/contamination, residues, smudges, print defects, bubbles/delamination, scratches or scuffs
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical anomalies

## Tone
    - Neutral and objective
'''

anomalies_fryum_change_criteria = '''
## Change to Detect
    - Break: fractured, chipped, or missing sections of the fryum structure
    - Overlap: multiple fryums stacked, overlapped, or fused together
    - Stain: visible contamination, discoloration, or foreign residue on the surface
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_grid_change_criteria = '''
## Change to Detect
    - Bent: deformation or warping of the grid structure or mesh wires
    - Broken: fractured, snapped, or missing segments of the grid
    - Glue: presence of adhesive residues or unintended glue marks
    - Metal contamination: embedded or attached foreign metallic particles
    - Thread anomaly: misaligned, loosened, or missing threads in the grid pattern
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_gum_change_criteria = '''
## Change to Detect
    - Stain: visible contamination, discoloration, or foreign residue on the gum surface
    - Surface damage: cracks, scratches, dents, or deformation on the gum texture or shape
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_hazelnut_change_criteria = '''
## Change to Detect
    - Crack: visible fractures or splits on the nut surface
    - Cut: sliced, chipped, or partially removed sections of the shell
    - Hole: perforations or missing portions on the nut body
    - Print: unintended imprints, marks, or text defects on the surface
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_leather_change_criteria = '''
## Change to Detect
    - Color anomaly: discoloration, uneven tones, or irregular color patches on the leather surface
    - Cut: sliced, torn, or partially removed sections of the material
    - Fold: unwanted creases, wrinkles, or bent areas altering the surface flatness
    - Glue: visible adhesive marks, residues, or unintended glossy spots
    - Poke: punctures, holes, or sharp indentations on the surface
    - Recovery or restoration from a previously damaged state

## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_macaroni_change_criteria = '''
## Change to Detect
    - Contamination: stains, foreign substances, or discoloration on the macaroni surface
    - Crack: visible fractures or splits along the pasta body
    - Hole: irregular or missing internal openings differing from normal tube structure
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_metal_nut_change_criteria = '''
## Change to Detect
    - Bent: deformation or warping of the nut shape or thread structure
    - Color anomaly: discoloration, oxidation, or uneven metallic tone
    - Flip: nut orientation changed, inverted, or rotated from its original alignment
    - Scratch: visible abrasions, scuffs, or surface wear on the metal surface
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_pcb_change_criteria = '''
## Change to Detect
    - Bend: warping, twisting, or deformation of the PCB board or components
    - Stain: visible contamination, discoloration, or residue on the board surface
    - Surface damage: scratches, cracks, dents, or chipped areas on the PCB or soldered parts
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_pill_change_criteria = '''
## Change to Detect
    - Color anomaly: discoloration, uneven tone, or incorrect color variation on the pill surface
    - Contamination: stains, foreign substances, or residue on the pill or coating
    - Crack: visible fractures or splits on the pill body
    - Faulty imprint: misprinted, blurred, or missing text/logo markings
    - Pill type: shape, size, or pattern differing from the expected type
    - Scratch: abrasions, scuffs, or surface wear marks
    - Combined defects: any co-occurrence of the above anomaly types
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_pipe_fryum_change_criteria = '''
## Change to Detect
    - Overlap: multiple fryums stacked, overlapped, or fused together
    - Stain: visible contamination, discoloration, or foreign residue on the surface
    - Surface damage: cracks, scratches, dents, or broken edges on the fryum body
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_screw_change_criteria = '''
## Change to Detect
    - Manipulated front: deformation, flattening, or irregular shaping of the screw head front
    - Scratch (head/neck): visible abrasions, scuffs, or wear marks on the screw head or neck area
    - Thread (side/top): damage, bending, flattening, or missing segments on the screw thread region
    - Combined defects: co-occurrence of any of the above anomaly types
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_tile_change_criteria = '''
## Change to Detect
    - Crack: visible fractures, splits, or chipped sections on the tile surface
    - Glue strip: adhesive residues, streaks, or unintended glue lines
    - Gray stroke: irregular gray marks or stains differing from the original pattern
    - Oil: oily smudges, stains, or reflective contamination on the surface
    - Rough: uneven texture, abrasion, or worn-out surface areas indicating material degradation
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_toothbrush_change_criteria = '''
## Change to Detect
    - Defective toothbrush: broken, bent, or deformed bristles or handle
    - Missing or misaligned bristles
    - Surface contamination, discoloration, or material damage on the brush head or grip area
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_transistor_change_criteria = '''
## Change to Detect
    - Bent lead: deformation or bending of one or more transistor leads
    - Cut lead: broken, shortened, or missing leads
    - Damaged case: cracks, scratches, dents, or chipped sections on the transistor body
    - Misplaced: incorrect position, tilt, or rotation of the transistor component
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_wood_change_criteria = '''
## Change to Detect
    - Color anomaly: discoloration, uneven tone, or irregular color patches on the wood surface
    - Combined defect: co-occurrence of multiple anomaly types such as cracks, stains, or scratches
    - Hole: drilled, missing, or irregular openings in the wood body
    - Liquid: stains, spills, or absorbed moisture marks on the surface
    - Scratch: visible abrasions, scuffs, or surface wear on the material texture
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

anomalies_zipper_change_criteria = '''
## Change to Detect
    - Broken teeth: missing, cracked, or deformed zipper teeth
    - Split or squeezed teeth: teeth misaligned, separated, or compressed abnormally
    - Fabric border: frayed, torn, or displaced fabric along the zipper edge
    - Fabric interior: damage, folds, or misplacement of the inner textile near the zipper
    - Rough surface: abrasions, uneven texture, or material wear
    - Combined defect: co-occurrence of multiple anomaly types above
    - Recovery or restoration from a previously damaged state
    
## No Change
    - Lighting or exposure variations
    - Viewpoint or camera shifts
    - Minor photometric, reflection, or compression artifacts
    - Color tone or shading differences not caused by physical defects

## Tone
    - Neutral and objective
'''

image_editing_change_criteria = '''
## CHANGE TO DETECT
- Modification or replacment of the background while preserving the main subject.
- Bidirectional style transformation (e.g., watercolor, sketch, realistic) while keeping the original structure and semantics intact.
- Object-level Modification:
    - Addition: introduce a new object 
    - Removal: eliminate an existing object and restore the background
    - Replacement: substitute one object with another while maintaining spatial consistency 

## NO CHANGE
- Non-intentional or minor variations not representing an explicit editing operation.

## Tone
    - Instructional and imperative (e.g., “Transform the image into a watercolor painting style.”)
'''

satellite_imagery_building_changes_change_criteria = '''
## Change to Detect
    - Construction or appearance of new buildings or built structures
    - Demolition, removal, or disappearance of existing buildings
    - Structural modifications such as expansion, extension, roof change, or reconstruction
    - Layout or footprint alterations indicating new urban development

## No Change
    - Lighting, shadow, or seasonal variations
    - Viewpoint or satellite orbit differences
    - Minor vegetation or texture changes not associated with building structures
    - Temporary objects (vehicles, cranes, materials) not part of permanent buildings

## Tone
    - Neutral and objective
'''

satellite_imagery_disaster_impact_change_criteria = '''
## Change to Detect
    - Physical destruction or collapse of buildings, bridges, or infrastructure caused by disasters (e.g., earthquake, warfare, wind)
    - Burnt, flooded, or buried regions indicating fire, flood, landslide, or volcanic impact
    - Erosion, sediment displacement, or debris accumulation caused by tsunami, flood, or landslide
    - Formation of new surface features such as ash deposits, lava flow, or landslide scars
    - Large-scale environmental changes resulting from disaster impact (e.g., scorched vegetation, inundated areas)
    - Recovery or restoration from previously damaged or impacted states
    
## No Change
    - Lighting, shadow, or seasonal differences between pre- and post-event images
    - Atmospheric variations such as cloud coverage or haze
    - Minor texture or color differences unrelated to structural or surface damage
    - Temporary vehicles, tents, or equipment not indicative of long-term disaster impact

## Tone
    - Neutral and objective
'''

satellite_imagery_residential_zoning_change_criteria = '''
## Change to Detect
    - Development of new residential areas or housing complexes in previously undeveloped land
    - Expansion or densification of existing residential zones
    - Conversion of non-residential areas (e.g., farmland, forest, open land) into housing or urban layouts
    - Structural reconfiguration of neighborhood layouts indicating planned urban growth
    - Reduction, removal, or reversion of residential development
 
## No Change
    - Lighting, shadow, or seasonal variations
    - Temporary construction equipment, vehicles, or materials not representing permanent development
    - Viewpoint or satellite orbit differences
    - Minor vegetation or soil color changes unrelated to zoning or land-use transformation

## Tone
    - Neutral and objective
'''

satellite_imagery_resource_extraction_change_criteria = '''
## Change to Detect
    - Expansion or reduction of mining areas, excavation zones, or quarry sites
    - Appearance or disappearance of exposed soil, pits, or extraction regions
    - Formation or enlargement of tailing ponds, waste piles, or material stockyards
    - Surface texture or color changes indicating active extraction or land degradation
    - Restoration, backfilling, or natural recovery of previously extracted or degraded areas
    
## No Change
    - Lighting, shadow, or seasonal variations
    - Viewpoint or satellite orbit differences
    - Temporary vehicles, machinery, or dust patterns not indicating structural land change
    - Minor vegetation or soil tone variations unrelated to mining activity

## Tone
    - Neutral and objective
'''

satellite_imagery_road_development_change_criteria = '''
## Change to Detect
    - Construction or appearance of new roads, highways, or paved routes
    - Expansion, extension, or rerouting of existing road networks
    - Conversion of unpaved roads into paved surfaces
    - Visible changes in road geometry or connectivity indicating infrastructure development
    - Reduction, removal, degradation, or disappearance of existing roads or paved surfaces
 
## No Change
    - Lighting, shadow, or seasonal variations
    - Temporary vehicles, construction machinery, or materials not representing permanent road changes
    - Viewpoint or satellite orbit differences
    - Minor vegetation or soil tone variations unrelated to actual road construction

## Tone
    - Neutral and objective
'''

satellite_imagery_sea_construction_change_criteria = '''
## Change to Detect
    - Construction or expansion of ports, piers, docks, or coastal facilities
    - Formation or extension of breakwaters, seawalls, or artificial islands
    - Installation or enlargement of offshore structures (e.g., oil platforms, wind turbines)
    - Reclamation activities or new landmasses formed along the shoreline
    - Movement, relocation, or disappearance/appearance of ships or large vessels
    - Reduction, removal, collapse, or erosion of existing coastal or offshore structures
   
## No Change
    - Lighting, shadow, or tidal variations
    - Ocean surface changes caused by waves, currents, or sediment drift
    - Cloud coverage, atmospheric haze, or viewpoint differences
    - Minor color or texture variations in water not related to structural development

## Tone
    - Neutral and objective
'''


satellite_imagery_water_dynamics_change_criteria = '''
## Change to Detect
    - Expansion, contraction, or displacement of water bodies such as rivers, lakes, or reservoirs
    - Formation or disappearance of ponds, floodplains, or wetlands
    - Shifts in shorelines or river courses indicating hydrological variation
    - Changes in water extent due to flooding, drought, or dam operation

## No Change
    - Lighting, shadow, or seasonal color variations of water
    - Tidal or wave pattern fluctuations without persistent area change
    - Cloud coverage, atmospheric haze, or viewpoint differences
    - Minor texture or reflection differences not related to water extent changes

## Tone
    - Neutral and objective
'''


inference_template = '''
You are an advanced vision–language reasoning model specialized in change captioning.
Your task is to analyze two semantically related images—image_t0 (earlier state) and image_t1 (later state)—and produce an accurate textual description of the visual changes **from image_t0 to image_t1**.

You will be given a **Change Criteria**, specifying which changes to detect and which variations to ignore. Follow it strictly.

# Change Criteria:
[change_criteria]

# Instructions:

1. Identify only the changes that satisfy the given **Change Criteria**.
2. If no valid change exists, output exactly:
    <output>
    There is no identifiable change.
    </output>

3. Ensure clarity and precision:
    - Use concise and domain-appropriate language.
    - Avoid speculative terms (e.g., “maybe,” “appears,” “seems”).

4. Base your description solely on observable evidence:
    - Describe what changed, where it occurred, and how it evolved **from t0 to t1**.

5. Output requirements (STRICT):
    - Do not mention or imply unchanged regions or aspects.
    - Write a single coherent paragraph (no lists or numbering).
    - Do not reference image ordering (e.g., “first image,” “second image”).
    - Wrap the final description in:
        <output>
        [Concise description of the detected change(s).]
        </output>

# Image Pairs:
[image_order_instruction]
'''

# Image-ordering instruction blocks injected into the templates above.
# 'no_tag' leaves ordering implicit; 'tag' names each image's temporal role.
no_tag = """
"""

tag = """
Each image is preceded by a textual tag indicating its role.
- An image following “[image_t0]” is the earlier state.
- An image following “[image_t1]” is the later state.
"""

inference_template_naive = '''
You are an advanced vision–language reasoning model specialized in change captioning.
Your task is to analyze two semantically related images—image_t0 (earlier state) and image_t1 (later state)—and produce an accurate textual description of the visual changes **from image_t0 to image_t1**.

# Instructions:

1. If no valid change exists, output exactly:
    <output>
    There is no identifiable change.
    </output>

2. Ensure clarity and precision:
    - Use concise and domain-appropriate language.
    - Avoid speculative terms (e.g., “maybe,” “appears,” “seems”).

3. Base your description solely on observable evidence:
    - Describe what changed, where it occurred, and how it evolved **from t0 to t1**.

4. Output requirements (STRICT):
    - Do not mention or imply unchanged regions or aspects.
    - Write a single coherent paragraph (no lists or numbering).
    - Do not reference image ordering (e.g., “first image,” “second image”).
    - Wrap the final description in:
        <output>
        [Concise description of the detected change(s).]
        </output>

# Image Pairs:
[image_order_instruction]
'''


evaluate_template = '''
You are an impartial judge assessing the quality of a predicted change caption in comparison to the ground-truth reference.

# Evaluation Criteria:

You evaluate the quality of the output answer following 4 criteria.

1. **Correctness (1–10)**: Evaluate how accurately the caption represents the actual visual changes with respect to the ground-truth reference. Assess whether objects, attributes, spatial relations, and transformation types precisely match the reference, without introducing contradictions, omissions, or misinterpretations.

2. **Specificity (1–10)**: Evaluate how detailed and precise the caption is relative to the ground-truth reference, emphasizing whether it preserves the same level of specificity in object categories, attributes, and spatial relations.

3. **Relevance (1–10)**: Evaluate whether the caption focuses exclusively on the meaningful change described in the ground-truth reference. Penalize any mention of unchanged elements or hallucinated additions that are not present in the reference transformation.

4. **Fluency (1–10)**: Evaluate the tone consistency and naturalness of the caption in comparison to the ground-truth reference. If the sentence mood differs—such as an imperative or directive form versus a neutral, descriptive statement—treat this as a major fluency error.
    -   Example:       
        Reference: "the car on the road has disappeared (descriptive)."
        Prediction: "remove the car on the road (imperative)."
        → tone mismatch; very low fluency.
            
# Instructions:

1.  Ensure that your evaluations are unbiased and based solely on the criteria provided.

2.  After evaluating all criteria, assign only a single final, human-aligned score that reflects the overall quality of the caption.

3.  **Output format:**
    -   Return a single score wrapped in:
        ```
        <correctness>
        [a score from 1 to 10]
        </correctness>

        <specificity>
        [a score from 1 to 10]
        </specificity>

        <relevance>
        [a score from 1 to 10]
        </relevance>

        <fluency>
        [a score from 1 to 10]
        </fluency>

        <final_score>
        [a score from 1 to 10]
        </final_score>
        ```

# Prediction:
[prediction]

# Reference:
[reference]
'''

# ---------------------------------------------------------------------------
# Programmatic access
# ---------------------------------------------------------------------------

#: Catalog of change criteria keyed by ``"<domain>"`` or ``"<domain>_<topic>"``,
#: auto-collected from the ``*_change_criteria`` constants defined above.
CHANGE_CRITERIA: dict[str, str] = {
    name[: -len("_change_criteria")]: value
    for name, value in dict(globals()).items()
    if name.endswith("_change_criteria") and isinstance(value, str)
}


def _slug(text: str) -> str:
    """Normalise free text to a lowercase, underscore-delimited lookup key."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def resolve_change_criteria(domain: str, topic: str) -> str:
    """Resolve change criteria with ``domain_topic`` -> ``domain`` -> ``generic`` fallback.

    Args:
        domain: high-level domain, e.g. ``"natural_scenes"``.
        topic: fine-grained topic, e.g. ``"meeting_room"``.

    Returns:
        The matching criteria text.

    Raises:
        KeyError: if no criteria are registered for the domain/topic.
    """
    dom, top = _slug(domain), _slug(topic)
    for key in (f"{dom}_{top}", dom, "generic"):
        if key in CHANGE_CRITERIA:
            return CHANGE_CRITERIA[key]
    raise KeyError(f"No change criteria registered for domain={domain!r}, topic={topic!r}.")


def build_inference_prompt(
    domain: str,
    topic: str,
    *,
    use_criteria: bool = True,
    tagged: bool = True,
) -> str:
    """Assemble the change-captioning prompt for an image pair.

    Args:
        domain, topic: used to look up domain-specific change criteria.
        use_criteria: if ``True`` use the criteria-guided template, otherwise the
            naive template (which carries no change criteria).
        tagged: if ``True`` the images are introduced with ``[image_t0]`` /
            ``[image_t1]`` text tags; otherwise their order is left implicit.
    """
    template = inference_template if use_criteria else inference_template_naive
    template = template.replace("[image_order_instruction]", tag if tagged else no_tag)
    if "[change_criteria]" in template:
        template = template.replace("[change_criteria]", resolve_change_criteria(domain, topic))
    return template


def build_eval_prompt(prediction: str, reference: str) -> str:
    """Fill the LLM-judge template with a prediction and its reference caption."""
    return evaluate_template.replace("[prediction]", prediction).replace("[reference]", reference)
