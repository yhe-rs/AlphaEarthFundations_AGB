import os
from tqdm import tqdm
from datetime import datetime, timedelta
import ee

ee.Authenticate()
ee.Initialize(project='fates-mrv')


if __name__ == "__main__":
    
    # define chunk size
    CHUNK_SIZE = 1 # per cell per chunk
    
    # gee assets path
    parent ='projects/fates-mrv/assets'
    
    # Ask GEE for the list of *direct* children of that folder
    resp = ee.data.listAssets({'parent': parent})  # dict with key 'assets'
    assets_meta = resp.get('assets', [])           # list of dictionaries
    
    # Keep only assets whose type == 'TABLE'   (vectors/shapefiles)
    table_ids = [a['name'] for a in assets_meta if a['type'] == 'TABLE']
    print('Found', len(table_ids), 'vector assets')
    
    # Print all found asset IDs with index
    for i, aid in enumerate(table_ids):
        print(f'{i:2d}. {aid}')
   
    year = 2019    
    START_DATE = f'{year}-01-01'
    END_DATE   = f'{year}-12-31'

    
    #  Read them in a loop
    for asset_id in tqdm(table_ids):
        #print(table_ids)

        # Filter to process only assets
        if not asset_id.endswith(f'{year}'):
            continue  # Skip if it does not end with 'fishnet'

        # Extract basename from asset ID for naming outputs
        basename = os.path.basename(asset_id)      # ← fishnet
        print('\n───────────────────────────────────────────────')
        print('Processing:', asset_id)

        # Load the vector asset as a FeatureCollection
        fishnet = ee.FeatureCollection(asset_id)
        # Get number of features (grid cells) in the collection (client-side)
        total_cells = fishnet.size().getInfo()
        print('\nNumber of grid cells :', total_cells)
        print('Schema columns:', fishnet.first().propertyNames().getInfo())

        # spatial tiling : loop over the grid in chunks
        # for offset in range(0, total_cells, CHUNK_SIZE):
        for offset in [4958]:#range(0, total_cells, CHUNK_SIZE):
            chunk_idx = offset // CHUNK_SIZE
            chunk_fc  = ee.FeatureCollection(fishnet.toList(CHUNK_SIZE, offset))
            
            # Extract geometry of chunk_fc
            chunk_geom = chunk_fc.geometry()

            # Filter the GLO-30 DEM ImageCollection to images intersecting the polygon geometry
            alpha_collection = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                               .filterBounds(chunk_geom)  # geom touched scene
                                .filterDate(START_DATE, END_DATE)
                               .select(['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09',
                                        'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
                                        'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 
                                        'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39',
                                        'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 
                                        'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 
                                        'A60', 'A61', 'A62', 'A63'])
                               .map(lambda image: image.toFloat())
                               # .map(lambda image: image.toDouble())
                              )
            print('Bands info:', alpha_collection.first().bandNames().getInfo())

            # Check bands before reduction: get the number of images in the filtered collection (client-side)
            collection_size = alpha_collection.size().getInfo()
            
            if collection_size == 0:
                 # No images found for this polygon, skip processing
                print(f"No images found for chunk {chunk_idx}. Skipping...")
                continue  # skip to next chunk


            else:
                print(f"⚠️  alpha collection_size: {collection_size}")
                # Map over the ImageCollection to perform reduceRegions on each image
                # and add a 'date' property to each resulting feature
                def addDateAndReduce(image):
                    # Perform reduceRegions on the current image
                    reduced_features = image.reduceRegions(
                        collection=chunk_fc,  # The chunk of features
                        reducer=ee.Reducer.mean(),  # Mean per feature (polygon)
                        scale=10,  # Resolution,
                        crs='EPSG:3857',
                        tileScale=1
                    )
                    # Add the date as a property to each feature
                    return reduced_features.map(lambda feature: feature.set('date', image.date().format('YYYY-MM-dd')))
                
                # Apply the function to each image in the collection and flatten the result
                alpha_samples = (alpha_collection.map(lambda image: image)  # Fix: Map clip to each image
                                                .map(addDateAndReduce)
                                                .flatten())
                
                print('⚠️  alpha samples size:', alpha_samples.size().getInfo())

                # # Print the features in alpha_sampless
                # features = alpha_samples.toList(alpha_samples.size())
                # feature_count = features.size().getInfo()
                # for i in range(feature_count):
                #     feature = ee.Feature(features.get(i))
                #     print(feature.getInfo())

                
                # Create a descriptive task name for export
                task_desc = f'alpha_{basename}_chunk_{chunk_idx}'
    
                # Set up and start an export task to Google Drive
                task = ee.batch.Export.table.toDrive(**{        
                    'collection': alpha_samples,
                    'description': task_desc, # set file name
                    # 'folder': 'alpha',  # folder on google drive
                    'fileFormat': 'csv'  # format
                })
    
                task.start()
                
                print(f'\n    → task: {task_desc} submitted '
                      f'({offset:,} – {min(offset+CHUNK_SIZE, total_cells)-1:,})')
                    
                # Optional: sleep to avoid submitting too many tasks too quickly
                # Each project's queue supports a maximum of 3,000 tasks
                import time
                time.sleep(3) # sleep interval to aviod creating multi outdir in google drive
                # do a pre-check of time/chunk to set this sleep value
