CREATE PROCEDURE [dbo].[HRMS_GetLocation.1.0.1]
	
AS
BEGIN 
	SET NOCOUNT ON

	SELECT 
		[Id], 
		[LocationName], 
		[Latitude], 
		[Longitude], 
		[Range],
		[IsEnabled]
	FROM [dbo].[LocationDetail] WITH(NOLOCK)
END