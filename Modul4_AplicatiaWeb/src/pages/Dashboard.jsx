import React, { useRef, useEffect, useState } from 'react';

// import Datepicker from '../partials/actions/Datepicker';
import PositiveOverTime from '../partials/dashboard/PositiveOverTime';
import SentimentOverTime from '../partials/dashboard/SentimentOverTime';
import DashboardCard11 from '../partials/dashboard/trash/DashboardCard11';
import MongoService from '../utils/MongoService';
import SearchComponent from '../partials/dashboard/SearchComponent';
import Specifications from '../partials/dashboard/SpecificationsComponents/Specifications';
import RecommendationContainer from '../partials/dashboard/recommendations/RecommendationContainer';

function Dashboard() {
  const [apiData, setApiData] = useState(null)
  const [link, setLink] = useState("");
  const [clientDb, setClient] = useState(null);
  const [error, setError] = useState("");

  const handleRecommendationClick = (link) => {
    setLink(link.toString());
  }

  const handleSearchSubmit = (searchValue) => {
    setLink(searchValue);
  };

  useEffect(() => {
    if (link != "") {
      let mongo = new MongoService(link);
      mongo.getLaptop().then((response) => {
        if (response.hasOwnProperty("valid")) {
          setApiData(response["valid"]);
          setClient(mongo);
          setError("");
        }
        else if (response.hasOwnProperty("error")) {
          if (response["error"] === 404) {
            setError("Link-ul este invalid sau nu avem informații despre produsul dorit!")
          }
        }
      });
    }
  }, [link])

  return (
    <>
      <div className="flex h-screen overflow-hidden">
        <div className="relative flex flex-col flex-1 overflow-y-auto overflow-x-hidden">
          <main>
            <div className="px-4 sm:px-6 lg:px-8 py-8 w-full max-w-screen-2xl mx-auto">
              {/*Componenta de cautare*/}
              <SearchComponent onSubmit={handleSearchSubmit} currentLink={link} error={error} />

              {apiData && <Specifications specData={apiData} />}
              <div className="grid grid-cols-12 gap-6">
                {/* Evolutia gradului de multumire in timp*/}
                {apiData && <SentimentOverTime  mongoData={apiData} />}
                {/* Frecventa polaritatii recenziilor */}
                {apiData && <PositiveOverTime  mongoData={apiData} />}
                {/* Recomandari si produse similare */}
                {(apiData && clientDb) && <RecommendationContainer service={clientDb} linkHandle={handleRecommendationClick} />}

              </div>

            </div>
          </main>
        </div>
      </div>
    </>
  );
}

export default Dashboard;