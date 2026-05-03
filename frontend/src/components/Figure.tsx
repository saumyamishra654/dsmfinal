interface FigureProps {
  src: string;
  alt: string;
  caption?: string;
}

export default function Figure({ src, alt, caption }: FigureProps) {
  return (
    <figure className="my-8 not-prose">
      <div className="rounded-xl overflow-hidden shadow-lg border border-gray-100 bg-white">
        <img
          src={src}
          alt={alt}
          className="w-full h-auto"
          loading="lazy"
        />
      </div>
      {caption && (
        <figcaption className="mt-3 text-center text-sm text-gray-500 italic px-4">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
