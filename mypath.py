class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return 'G:/VOC - dataset/VOCdevkit/VOC2012'   # folder that contains VOCdevkit/.
        elif database == 'sbd':
            return 'G:/VOC - dataset/BSDS500'  # folder that contains dataset/.
        elif database == 'coco':
            return '/path/to/coco'  # folder that contains annotations/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return '/path/to/Models/'