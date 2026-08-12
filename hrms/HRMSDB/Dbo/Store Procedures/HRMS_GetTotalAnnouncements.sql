CREATE PROCEDURE [dbo].[HRMS_GetTotalAnnouncements.1.0.0]
AS
BEGIN 
	SET NOCOUNT ON

	SELECT COUNT([Id]) FROM [dbo].[Announcement] WITH(NOLOCK);
END