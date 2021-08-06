import os
import random as rnd

from PIL import Image, ImageFilter

from trdg import computer_text_generator, background_generator, distorsion_generator

try:
    from trdg import handwritten_text_generator
except ImportError as e:
    print("Missing modules for handwritten text generation.")


class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(
            cls,
            index,
            text,
            font,
            out_dir,
            size,
            extension,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            is_handwritten,
            name_format,
            width,
            alignment,
            text_color,
            orientation,
            space_width,
            character_spacing,
            margins,
            fit,
            output_mask,
            word_split,
            image_dir,
            stroke_width=0,
            stroke_fill="#282828",
            image_mode="RGB",
    ):
        """

        Args:
            index(int): the index of ith generated text
            text(str): the content of text
            font(str): the ttf file with corresponding font type.
            out_dir(str): output dir
            size(int): 生成的背景高度(水平文本)/宽度(垂直文本) 字体尺寸
            extension(str): the suffix of image
            skewing_angle(int): 歪斜角度
            random_skew(bool): If true, the skewing angle would be [-a,a]
            blur(int): 模糊半径/强度
            random_blur(bool):
            background_type(int): 0:gaussian noise 1:plain white 2: 晶体 3: 图片
            distorsion_type(int): 0: not distortion, 1: sin distortion, 2: cos distortion, else: random distortion
            distorsion_orientation(int): 0: vertical 1: horizontal 2: both
            is_handwritten(bool): Whther use GAN to generate handwritten text
            name_format(int):produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings
            width(int): the background width
            alignment(int): 0: left alignment, 1: center alignment, 2: right alignment
            text_color(str): text color
            orientation(int): 0 is horizontal, 1 is vertical
            space_width(float): 单词之间的空格宽度,这里表示" "宽度的倍数
            character_spacing(int): 字符之间的空格宽度，像素为单位，Default is 0
            margins(tuple(int)): the distance between four directions and text region. 一般实际字体大小= 字体尺寸- margins方向
            fit(bool): Whether crop the text region to make the image looks tight.
            output_mask(bool): Whether output text region mask
            word_split(bool):
            image_dir(str): the background image dir
            stroke_width(int): 笔画宽度
            stroke_fill(str): stroke color filled in
            image_mode(str): Default is 'RGB'

        Returns:

        """
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image, mask = handwritten_text_generator.generate(text, text_color)
        else:
            try:
                image, mask, t_color = computer_text_generator.generate(
                    text,
                    font,
                    text_color,
                    size,
                    orientation,
                    space_width,
                    character_spacing,
                    fit,
                    word_split,
                    stroke_width,
                    stroke_fill,
                )
            except Exception as e:
                return None
        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        ##################################
        # Apply distorsion to text image #
        ##################################
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(
                distorted_img.size[0]
                * (float(size - vertical_margin) / float(distorted_img.size[1]))
            )
            resized_img = distorted_img.resize(
                (new_width, size - vertical_margin), Image.ANTIALIAS
            )
            resized_mask = distorted_mask.resize((new_width, size - vertical_margin), Image.NEAREST)
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_img.size[1])
                * (float(size - horizontal_margin) / float(distorted_img.size[0]))
            )
            resized_img = distorted_img.resize(
                (size - horizontal_margin, new_height), Image.ANTIALIAS
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.NEAREST
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )
        elif background_type == 3:
            background_img = background_generator.plain_color(
                background_height, background_width, t_color
            )
        else:
            background_img = background_generator.image(
                background_height, background_width, image_dir, t_color
            )
        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background_img.paste(resized_img, (margin_left, margin_top), resized_img)
            background_mask.paste(resized_mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (int(background_width / 2 - new_text_width / 2), margin_top),
            )
        else:
            background_img.paste(
                resized_img,
                (background_width - new_text_width - margin_right, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (background_width - new_text_width - margin_right, margin_top),
            )

        #######################
        # Apply gaussian blur #
        #######################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.randint(0, blur)
        )
        final_image = background_img.filter(gaussian_filter)
        final_mask = background_mask.filter(gaussian_filter)

        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################

        final_image = final_image.convert(image_mode)
        final_mask = final_mask.convert(image_mode)

        # rotate the vertical image
        if orientation == 1:
            final_image = final_image.rotate(90, expand=True)
            final_mask = final_mask.rotate(90, expand=True)

        #####################################
        # Generate name for resulting image #
        #####################################
        # We remove spaces if space_width == 0
        if space_width == 0:
            text = text.replace(" ", "")
        if name_format == 0:
            image_name = "{}_{}.{}".format(text, str(index), extension)
            mask_name = "{}_{}_mask.png".format(text, str(index))
        elif name_format == 1:
            image_name = "{}_{}.{}".format(str(index), text, extension)
            mask_name = "{}_{}_mask.png".format(str(index), text)
        elif name_format == 2:
            image_name = "{}.{}".format(str(index), extension)
            mask_name = "{}_mask.png".format(str(index))
        else:
            print("{} is not a valid name format. Using default.".format(name_format))
            image_name = "{}_{}.{}".format(text, str(index), extension)
            mask_name = "{}_{}_mask.png".format(text, str(index))

        # Save the image
        if out_dir is not None:
            final_image.save(os.path.join(out_dir, image_name))
            if output_mask == 1:
                final_mask.save(os.path.join(out_dir, mask_name))
        else:
            if output_mask == 1:
                return final_image, final_mask
            return final_image
