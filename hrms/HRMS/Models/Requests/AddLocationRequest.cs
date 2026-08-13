using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class AddLocationRequest : LocationDetails
    {
        [JsonIgnore]
        public override int Id { get; set; }

        [JsonIgnore]
        public override bool IsEnabled { get; set; }
    }
}
