import React, { useState, useEffect, useRef } from 'react';

function SearchComponent(props) {
  const [searchValue, setValue]= useState("");
  const searchRef = useRef();
  const [error, setError] = useState("")

  const handleSubmit = (event) => {
    event.preventDefault();
    let link = searchValue;
    let validatedLink = "";
    let isLinkValid = false;
    validatedLink = validateLink(link)[0]
    isLinkValid = validateLink(link)[1]

    if (isLinkValid) {
      props.onSubmit(validatedLink);
      setError("")
    }
    else {
        setError("Introduceți un link valid!")
    }
  };

  useEffect(() => {
    searchRef.current.focus();
    setValue(props.currentLink)
    setError(props.error)
  }, [props]);

  const handleInputChange = (event) => {
    setValue(event.target.value)

  };

  const validateLink = (link) => {
    let validatedLink = link;
    let isValid = false;

    const emag_full_regex = /^https?:\/\/(?:www\.)?emag\.ro\/(?!.*\/{2,})[^\/]+\/pd\/\w+\//;
    const no_https_www_emag = /^emag\.ro\/(?:.+)\/pd\/\w+/;
    const no_https_emag = /^www\.emag\.ro\/(?:.+)\/pd\/\w+/;

    if (no_https_www_emag.test(link)) {
      validatedLink = "https://www." + link;
      isValid = true;
    }
    else if (no_https_emag.test(link)) {
      validatedLink = "https://" + link;
      isValid = true;
    }

    if (emag_full_regex.test(validatedLink)) {
      isValid = true;
    }
    else
      isValid = false;

    if (isValid) {
      validatedLink = validatedLink.match(/^(.+\/pd\/\w+\/)/)[1]
      if (validatedLink[validatedLink.length - 1] !== "/")
        validatedLink = validatedLink + "/";
    }

    return [validatedLink, isValid]
  }


  return (
      <div
        ref={searchRef}
        className="bg-white overflow-auto w-full max-h-full rounded shadow-lg mb-9"
      >
        <form className="border-b border-slate-200" onSubmit={handleSubmit}>
          <div className="relative">
            <input
              className="border-0 focus:ring-transparent placeholder-slate-400 appearance-none py-3 pl-10 pr-4 w-full max-h-full overflow-auto"
              type="search"
              placeholder="Introduce-ti link-ul"
              onChange={handleInputChange}
              value={searchValue}
              />
            <button
              className="absolute inset-0 right-auto group"
              type="submit"
              aria-label="Search"
              >
              <svg
                className="w-4 h-4 shrink-0 fill-current text-slate-400 group-hover:text-slate-500 ml-4 mr-2"
                viewBox="0 0 16 16"
                xmlns="http://www.w3.org/2000/svg"
                >
                <path d="M7 14c-3.86 0-7-3.14-7-7s3.14-7 7-7 7 3.14 7 7-3.14 7-7 7zM7 2C4.243 2 2 4.243 2 7s2.243 5 5 5 5-2.243 5-5-2.243-5-5-5z" />
                <path d="M15.707 14.293L13.314 11.9a8.019 8.019 0 01-1.414 1.414l2.393 2.393a.997.997 0 001.414 0 .999.999 0 000-1.414z" />
              </svg>
            </button>
          </div>
        </form>
        {error && <p className='font-semibold text-red-600 pl-4'>{error}</p>}
        </div>
  );
};


export default SearchComponent;